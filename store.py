"""Retain generated speech and original tool logs outside model context."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.audio = self.root / "audio"
        self.audio.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "voice.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS speech (
                id TEXT PRIMARY KEY, origin TEXT NOT NULL, text TEXT NOT NULL,
                audio_path TEXT NOT NULL, state TEXT NOT NULL,
                sent_at TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS tool_history_archive (
                id TEXT PRIMARY KEY, origin TEXT NOT NULL, cid TEXT NOT NULL,
                records TEXT NOT NULL, created_at REAL NOT NULL
            );
        """)

    def generated(self, speech_id, origin, text, path):
        with self.db:
            self.db.execute(
                "INSERT INTO speech (id,origin,text,audio_path,state) VALUES (?,?,?,?,?)",
                (speech_id, origin, text, str(path), "generated"),
            )

    def status(self, speech_id, state, sent_at="", error=""):
        with self.db:
            self.db.execute(
                "UPDATE speech SET state=?,sent_at=?,error=? WHERE id=?",
                (state, sent_at, error, speech_id),
            )

    def get(self, speech_id):
        row = self.db.execute(
            "SELECT * FROM speech WHERE id=?", (speech_id,)
        ).fetchone()
        return dict(row) if row else None

    def archive(self, origin, cid, history, entries):
        kept = dict(entries)
        records = [
            {"index": i, "message": message}
            for i, message in enumerate(history)
            if kept.get(i) != message
        ]
        if not records:
            return
        digest = hashlib.sha256(
            json.dumps(
                [origin, cid, history], ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO tool_history_archive VALUES (?,?,?,?,?)",
                (
                    digest,
                    origin,
                    cid,
                    json.dumps(records, ensure_ascii=False),
                    time.time(),
                ),
            )

    def close(self):
        self.db.close()
