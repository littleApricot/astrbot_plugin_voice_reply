"""Persist spoken transcripts through the same official context as compression."""

import asyncio
import copy
import json

TOOL = "voice_send"
SPEECH_HEADER = "[机器人已发送语音]"


def project(history):
    names = {}
    entries = []
    for index, raw in enumerate(history):
        message = copy.deepcopy(raw)
        if message.get("role") == "assistant" and message.get("tool_calls"):
            kept = []
            removed = False
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name")
                names[call.get("id")] = name
                if name == TOOL:
                    removed = True
                else:
                    kept.append(call)
            if removed:
                if kept:
                    message["tool_calls"] = kept
                else:
                    message.pop("tool_calls", None)
                    if isinstance(message.get("content"), list):
                        message["content"] = [
                            p for p in message["content"] if p.get("type") != "think"
                        ]
                    if not message.get("content"):
                        continue
        elif message.get("role") == "tool":
            if names.pop(message.get("tool_call_id"), None) == TOOL:
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                try:
                    data, _ = json.JSONDecoder().raw_decode(content.lstrip())
                except (ValueError, TypeError, AttributeError):
                    data = {}
                if (
                    isinstance(data, dict)
                    and data.get("ok") is True
                    and data.get("sent") is True
                    and isinstance(data.get("text"), str)
                    and data["text"].strip()
                ):
                    metadata = {
                        k: data[k] for k in ("speech_id", "sent_at") if k in data
                    }
                    entries.append(
                        (
                            index,
                            {
                                "role": "assistant",
                                "content": SPEECH_HEADER
                                + " "
                                + json.dumps(metadata, ensure_ascii=False)
                                + "\n"
                                + data["text"],
                            },
                        )
                    )
                continue
        entries.append((index, message))
    return entries


class HistoryWriter:
    """Transform writes only; preserve manager/read hooks owned by other plugins."""

    def __init__(self, database):
        self.database = database
        self.original = database.update_conversation
        self.owners = []
        self.pending = {}

        async def update(
            cid, title=None, persona_id=None, content=None, token_usage=None
        ):
            owner = next(
                (o for o in reversed(self.owners) if o.config.get("enabled", True)),
                None,
            )
            if owner is None or content is None:
                return await self.original(
                    cid,
                    title=title,
                    persona_id=persona_id,
                    content=content,
                    token_usage=token_usage,
                )
            task = asyncio.current_task()
            self.pending.setdefault(owner, set()).add(task)
            try:
                # No history read or plugin lock here: native database updates
                # may call the getter already wrapped by the meme integration.
                entries = project(content)
                cleaned = [message for _, message in entries]
                if cleaned != content:
                    owner.store.archive("", str(cid), content, entries)
                    token_usage = 0
                return await self.original(
                    cid,
                    title=title,
                    persona_id=persona_id,
                    content=cleaned,
                    token_usage=token_usage,
                )
            finally:
                self.pending.get(owner, set()).discard(task)

        self.wrapper = update
        database.update_conversation = update

    @classmethod
    def acquire(cls, owner):
        database = owner.context.conversation_manager.db
        writer = getattr(database, "_voice_history_writer", None)
        if writer is None:
            writer = cls(database)
            database._voice_history_writer = writer
        writer.owners.append(owner)
        return writer

    async def close(self, owner):
        if owner in self.owners:
            self.owners.remove(owner)
        tasks = self.pending.pop(owner, set()) - {asyncio.current_task()}
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if not self.owners and self.database.update_conversation is self.wrapper:
            self.database.update_conversation = self.original
            delattr(self.database, "_voice_history_writer")
