"""Let the reply model choose speech without enabling automatic TTS conversion."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import uuid
from pathlib import Path

from astrbot.api import AstrBotConfig, llm_tool
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.register import llm_tools

from .history import HistoryWriter
from .store import Store
from .synthesis import SERVICE_TEXT_LIMIT, HttpSpeechService, SpeechError, SpeechService

INSTRUCTIONS = """你可以根据当前对话自行选择文字或语音回复，不必每次发语音。
决定发语音时，用 voice_send(text) 传入你实际想说出的完整原文，不要附加工具执行汇报。
单次语音原文最多300个字符。长内容可以正常回复文字；需要多条语音时，按顺序在后续工具步骤发送，不要并行提交。
工具只发送语音，不会自动结束本轮；先读取发送结果，再决定下一步。
语音成功且已表达完毕、本轮有 silence 工具时，可以在下一步单独调用 silence，正文留空，结束本轮。
也可以在发语音后继续正常回复文字，但不要重复已经说过的同一句话。
失败不代表已经送达，送达状态不确定时不要重发同一条语音，可以用正常文字继续沟通。
后续历史上下文中标明已发送的语音原文，就是你之前实际说过的话。
"""


class VoiceReply(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config
        self.store = Store(StarTools.get_data_dir("astrbot_plugin_voice_reply"))
        self.service: SpeechService = HttpSpeechService(config)
        self.writer = HistoryWriter.acquire(self)
        tool = llm_tools.get_func("voice_send")
        if tool is not None:
            tool.parameters["required"] = ["text"]
            tool.parameters["properties"]["text"].update(
                minLength=1, maxLength=SERVICE_TEXT_LIMIT
            )
        self._active = set()
        self._closing = False

    @filter.on_llm_request(priority=9000)
    async def prepare(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("enabled", True) or not self.service.ready:
            if req.func_tool is not None:
                req.func_tool.remove_tool("voice_send")
            return
        req.system_prompt += "\n\n" + INSTRUCTIONS

    @llm_tool(name="voice_send")
    async def send(self, event: AstrMessageEvent, text: str) -> str:
        """选择用语音说话，合成并发送到当前对话，只负责发送语音。

        Args:
            text(string): 你想实际说出的完整原文，1至300个字符，不要包含工具执行汇报。

        Returns:
            ok、sent、语音原文及发送时间，或失败结果。确认发送成功后，可在后续一步单独调用 silence 结束本轮，也可继续回复文字。送达不确定时不要重发。
        """

        def result(**data):
            return json.dumps(data, ensure_ascii=False)

        if self._closing or not self.config.get("enabled", True):
            return result(ok=False, sent=False, error="disabled", retry_safe=False)
        if not self.service.ready:
            return result(
                ok=False,
                sent=False,
                error="speech_service_not_configured",
                retry_safe=False,
            )
        if not isinstance(text, str) or not text.strip():
            return result(ok=False, sent=False, error="empty_text", retry_safe=True)
        text = text.strip()
        text_limit = max(
            1,
            min(
                SERVICE_TEXT_LIMIT,
                int(self.config.get("max_text_chars", SERVICE_TEXT_LIMIT)),
            ),
        )
        if len(text) > text_limit:
            return result(ok=False, sent=False, error="text_too_long", retry_safe=True)
        key = hashlib.sha256(text.encode()).hexdigest()
        attempts = event.get_extra("voice_reply_attempts")
        if attempts is None:
            attempts = {}
            event.set_extra("voice_reply_attempts", attempts)
        if key in attempts:
            return result(
                ok=False,
                sent=False,
                error="already_attempted_this_turn",
                delivery=attempts[key],
                retry_safe=False,
            )
        attempts[key] = "generating"
        speech_id = uuid.uuid4().hex
        task = asyncio.current_task()
        self._active.add(task)
        sending = False
        generated = False
        try:
            audio = await self.service.synthesize(text, self.store.audio / speech_id)
            path = Path(audio.path).resolve()
            if (
                not path.is_relative_to(self.store.audio.resolve())
                or not path.is_file()
                or not path.stat().st_size
            ):
                raise SpeechError("invalid_audio_file")
            if not isinstance(audio.spoken_text, str) or not audio.spoken_text.strip():
                raise SpeechError("missing_spoken_text")
            self.store.generated(
                speech_id, event.unified_msg_origin, audio.spoken_text, path
            )
            generated = True
            self.store.status(speech_id, "sending")
            sending = True
            attempts[key] = "unknown"
            await event.send(
                MessageChain([Record.fromFileSystem(str(path), text=audio.spoken_text)])
            )
            sent_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
            attempts[key] = "sent"
            try:
                self.store.status(speech_id, "sent", sent_at=sent_at)
            except Exception as exc:
                self.logger.warning(
                    "Voice sent; receipt persistence failed: %s", type(exc).__name__
                )
            return result(
                ok=True,
                sent=True,
                speech_id=speech_id,
                text=audio.spoken_text,
                sent_at=sent_at,
            )
        except asyncio.CancelledError:
            if generated:
                self._record_failure(
                    speech_id, "unknown" if sending else "cancelled", "CancelledError"
                )
            if not sending:
                attempts.pop(key, None)
            raise
        except Exception as exc:
            error = str(exc) if isinstance(exc, SpeechError) else type(exc).__name__
            generation_unknown = error in (
                "tts_timeout_no_automatic_retry", "tts_network_error"
            ) or error.startswith("tts_http_5") and error != "tts_http_503"
            if generated:
                self._record_failure(
                    speech_id, "unknown" if sending else "failed", error
                )
            if not sending and not generation_unknown:
                attempts.pop(key, None)
            elif not sending:
                attempts[key] = "generation_unknown"
            return result(
                ok=False,
                sent=False,
                error=error,
                delivery="unknown" if sending else "not_sent",
                retry_safe=not sending and not generation_unknown,
            )
        finally:
            self._active.discard(task)

    def _record_failure(self, speech_id, state, error):
        try:
            self.store.status(speech_id, state, error=error)
        except Exception as exc:
            self.logger.warning(
                "Voice failure receipt unavailable: %s", type(exc).__name__
            )

    async def terminate(self):
        self._closing = True
        active = self._active - {asyncio.current_task()}
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        await self.writer.close(self)
        self.store.close()
