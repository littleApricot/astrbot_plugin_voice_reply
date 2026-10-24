"""Call the documented /tts endpoint and store its complete PCM WAV response."""

import asyncio
import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

SERVICE_TEXT_LIMIT = 300
MAX_AUDIO_BYTES = 64 * 1024 * 1024


class SpeechError(Exception):
    """A public error code that contains no endpoint or credential details."""


@dataclass(frozen=True)
class SpeechAudio:
    path: Path
    spoken_text: str


class SpeechService(Protocol):
    ready: bool

    async def synthesize(self, text: str, destination: Path) -> SpeechAudio:
        """Save audio in destination's directory and return its exact spoken text."""
        ...


class HttpSpeechService:
    """Serialize requests because the local model runs one generation at a time."""

    def __init__(self, config):
        self.config = config
        self._lock = asyncio.Lock()

    @property
    def ready(self):
        return bool(
            str(self.config.get("api_base_url") or "").strip()
            and str(self.config.get("api_key") or "").strip()
        )

    async def synthesize(self, text: str, destination: Path) -> SpeechAudio:
        if not self.ready:
            raise SpeechError("speech_service_not_configured")
        if not 1 <= len(text.strip()) <= SERVICE_TEXT_LIMIT:
            raise SpeechError("text_length_must_be_1_to_300")
        base = str(self.config.get("api_base_url", "")).strip().rstrip("/")
        try:
            url = httpx.URL(base)
        except httpx.InvalidURL as exc:
            raise SpeechError("invalid_service_url") from exc
        if (
            url.scheme not in ("https", "http")
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise SpeechError("invalid_service_url")
        endpoint = base if url.path.endswith("/tts") else base + "/tts"
        headers = {
            "Authorization": "Bearer " + str(self.config.get("api_key", "")).strip(),
            "Accept": "audio/wav",
        }
        timeout = max(1, int(self.config.get("synthesis_timeout_seconds", 120)))
        # A queued turn waits here, outside the timeout for an active generation.
        async with self._lock:
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=False, trust_env=False
                ) as client:
                    data = await asyncio.wait_for(
                        self._request(client, endpoint, headers, text), timeout=timeout
                    )
            except (httpx.TimeoutException, TimeoutError) as exc:
                raise SpeechError("tts_timeout_no_automatic_retry") from exc
            except httpx.HTTPError as exc:
                raise SpeechError("tts_network_error") from exc
        self._check_wav(data)
        path = destination.with_suffix(".wav")
        partial = path.with_suffix(".wav.part")
        try:
            partial.write_bytes(data)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        return SpeechAudio(path, text)

    async def _request(self, client, endpoint, headers, text):
        # Only a definite busy response is retried; timeouts and gateway errors
        # may leave a generation running on the server.
        for attempt in range(3):
            retry_delay = None
            async with client.stream(
                "POST", endpoint, headers=headers, json={"text": text}
            ) as response:
                if response.status_code == 503 and attempt < 2:
                    try:
                        retry_delay = max(0, int(response.headers.get("Retry-After", "3")))
                    except ValueError:
                        retry_delay = 3
                elif response.status_code != 200:
                    raise SpeechError(f"tts_http_{response.status_code}")
                else:
                    content_type = (
                        response.headers.get("Content-Type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
                        raise SpeechError("tts_response_is_not_wav")
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(chunks) + len(chunk) > MAX_AUDIO_BYTES:
                            raise SpeechError("tts_audio_too_large")
                        chunks.extend(chunk)
                    return bytes(chunks)
            if retry_delay is not None:
                await asyncio.sleep(retry_delay)
        raise SpeechError("tts_http_503")

    @staticmethod
    def _check_wav(data):
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise SpeechError("tts_invalid_wav")
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                if (
                    audio.getnchannels() != 1
                    or audio.getframerate() != 24000
                    or audio.getsampwidth() != 2
                    or audio.getcomptype() != "NONE"
                    or audio.getnframes() < 1
                ):
                    raise SpeechError("tts_unexpected_wav_format")
                expected = (
                    audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                )
                if len(audio.readframes(audio.getnframes())) != expected:
                    raise SpeechError("tts_incomplete_wav")
        except (wave.Error, EOFError) as exc:
            raise SpeechError("tts_invalid_wav") from exc
