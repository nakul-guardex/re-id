#!/usr/bin/env python3
"""
Session Event & Decision Logger
===============================
Persists every gate decision, appearance embedding, Hungarian match,
cannot-link constraint, and merge event to disk in JSONL format.
Enables offline replay, error debugging, and threshold calibration.
"""

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional
import numpy as np


class SessionEventLogger:
    """Thread-safe event logger recording pipeline decisions per session."""

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.current_session_id: Optional[str] = None
        self.log_file = None

    def start_session(self, session_id: str):
        with self.lock:
            self.close()
            self.current_session_id = session_id
            log_path = self.logs_dir / f"{session_id}_events.jsonl"
            self.log_file = open(log_path, "a", encoding="utf-8")
            self._write({
                "event": "SESSION_START",
                "session_id": session_id,
                "utc_timestamp": datetime.utcnow().isoformat(),
            })

    def log(self, event_type: str, data: Dict[str, Any]):
        """Logs an event payload to the JSONL log file."""
        with self.lock:
            if self.log_file is None:
                return
            entry = {
                "event": event_type,
                "session_id": self.current_session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            self._write(entry)

    def _write(self, obj: Dict[str, Any]):
        def _serialize(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.float32, np.float64)):
                return float(v)
            if isinstance(v, (np.int32, np.int64)):
                return int(v)
            return str(v)

        try:
            line = json.dumps(obj, default=_serialize) + "\n"
            self.log_file.write(line)
            self.log_file.flush()
        except Exception as ex:
            print(f"[EventLogger] Error writing event: {ex}")

    def close(self):
        with self.lock:
            if self.log_file:
                try:
                    self._write({"event": "SESSION_STOP", "session_id": self.current_session_id})
                    self.log_file.close()
                except Exception:
                    pass
                self.log_file = None
            self.current_session_id = None
