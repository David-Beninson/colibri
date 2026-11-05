"""Standalone session persistence and checkpointing store (`src/ai/session.py`).

Provides abstract session store interfaces and implementations for Colibri companion:
- `SessionStore`: Abstract base class for multi-turn session persistence.
- `SqliteSessionStore`: Zero-dependency SQLite checkpointer (`data/memory_db/checkpoints.sqlite`),
  compatible with both `ColibriAgent` and LangGraph `BaseCheckpointSaver`.
- `InMemorySessionStore`: Thread-safe in-memory session store for fast unit testing.
- `get_session_history`: Helper to retrieve message history for a given thread.
"""
from __future__ import annotations

import abc
import contextlib
import datetime
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    _LANGGRAPH_CP_AVAILABLE = True
except ImportError:
    class BaseCheckpointSaver:  # type: ignore
        pass

    class CheckpointTuple(NamedTuple):  # type: ignore
        config: Dict[str, Any]
        checkpoint: Dict[str, Any]
        metadata: Dict[str, Any]
        parent_config: Optional[Dict[str, Any]] = None

    _LANGGRAPH_CP_AVAILABLE = False


def _clean_messages_for_storage(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip static system prompt and transient `[LOCAL MEMORY CONTEXT]` suffixes before persisting."""
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        if role == "system":
            continue
        item = dict(m)
        if role == "user" and isinstance(item.get("content"), str):
            item["content"] = item["content"].split("\n\n[LOCAL MEMORY CONTEXT]", 1)[0]
        cleaned.append(item)
    return cleaned


def resolve_checkpoint_db_path(db_path: Optional[str] = None) -> str:
    """Resolve SQLite database path for session checkpoints.

    Priority:
    1. Explicit `db_path` argument.
    2. `COLIBRI_MEMORY_DIR` environment variable.
    3. `data/memory_db` relative to repository root.
    4. Fallback to `~/.colibri/memory` if repo directory is inaccessible.
    """
    if db_path:
        path = os.path.abspath(os.path.expanduser(db_path))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path

    env_dir = os.environ.get("COLIBRI_MEMORY_DIR")
    if env_dir:
        path = os.path.abspath(os.path.expanduser(env_dir))
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "checkpoints.sqlite")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repo_memory = os.path.join(repo_root, "data", "memory_db")
    try:
        os.makedirs(repo_memory, exist_ok=True)
        return os.path.join(repo_memory, "checkpoints.sqlite")
    except OSError:
        home_memory = os.path.expanduser("~/.colibri/memory")
        os.makedirs(home_memory, exist_ok=True)
        return os.path.join(home_memory, "checkpoints.sqlite")


class SessionStore(abc.ABC):
    """Abstract interface for session checkpoint persistence."""

    @abc.abstractmethod
    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_data: Union[Dict[str, Any], Sequence[Dict[str, Any]], None] = None,
        intent: str = "conversation",
        step_count: int = 0,
        *,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Persist checkpoint data or message sequence for a thread."""
        pass

    @abc.abstractmethod
    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint dict for `thread_id` or None if not found."""
        pass

    @abc.abstractmethod
    def get_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Return conversational message history for `thread_id`."""
        pass

    @abc.abstractmethod
    def clear(self, thread_id: str = "default") -> bool:
        """Clear checkpoint for `thread_id`. Return True if deleted, False otherwise."""
        pass

    @abc.abstractmethod
    def list_threads(self) -> List[str]:
        """Return list of distinct saved thread IDs."""
        pass


class SqliteSessionStore(SessionStore, BaseCheckpointSaver):
    """Persistent zero-dependency SQLite session store (`data/memory_db/checkpoints.sqlite`).

    Saves and restores multi-turn session history (`thread_id`, `messages`, `intent`,
    `step_count`, `updated_at`) across process restarts natively for both `ColibriAgent`
    and LangGraph `StateGraph`.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._explicit_db_path = db_path

    @property
    def db_path(self) -> str:
        return resolve_checkpoint_db_path(self._explicit_db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                thread_id TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL,
                intent TEXT NOT NULL DEFAULT 'conversation',
                step_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, task_id, channel)
            )
            """
        )
        return conn

    @staticmethod
    def _clean_messages_for_storage(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _clean_messages_for_storage(messages)

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_data: Union[Dict[str, Any], Sequence[Dict[str, Any]], None] = None,
        intent: str = "conversation",
        step_count: int = 0,
        *,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        tid = (thread_id or "default").strip() or "default"
        if messages is not None and checkpoint_data is None:
            checkpoint_data = messages
        if isinstance(checkpoint_data, dict):
            raw_messages = checkpoint_data.get("messages", [])
            intent = str(checkpoint_data.get("intent", intent))
            step_count = int(checkpoint_data.get("step_count", step_count))
        elif isinstance(checkpoint_data, (list, tuple)):
            raw_messages = checkpoint_data
        else:
            raw_messages = []
        clean_msgs = _clean_messages_for_storage(raw_messages)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO session_checkpoints (thread_id, messages_json, intent, step_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    intent = excluded.intent,
                    step_count = excluded.step_count,
                    updated_at = excluded.updated_at
                """,
                (tid, json.dumps(clean_msgs, ensure_ascii=False), intent, int(step_count), now_iso),
            )
            conn.commit()

    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        tid = (thread_id or "default").strip() or "default"
        if not os.path.isfile(self.db_path):
            return None
        try:
            with contextlib.closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints WHERE thread_id = ?",
                    (tid,),
                ).fetchone()
            if not row:
                return None
            return {
                "thread_id": row[0],
                "messages": json.loads(row[1]) if row[1] else [],
                "intent": row[2],
                "step_count": int(row[3] or 0),
                "updated_at": row[4],
            }
        except Exception:  # noqa: BLE001
            return None

    def get_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        cp = self.load_checkpoint(thread_id)
        if not cp:
            return []
        return list(cp.get("messages") or [])

    def clear(self, thread_id: str = "default") -> bool:
        tid = (thread_id or "default").strip() or "default"
        if not os.path.isfile(self.db_path):
            return False
        with contextlib.closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM session_checkpoints WHERE thread_id = ?", (tid,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (tid,))
            conn.commit()
            return cur.rowcount > 0

    def clear_checkpoint(self, thread_id: str) -> None:
        """Backwards compatibility alias for `clear()`."""
        self.clear(thread_id)

    def list_threads(self) -> List[str]:
        if not os.path.isfile(self.db_path):
            return []
        try:
            with contextlib.closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT thread_id FROM session_checkpoints ORDER BY updated_at DESC"
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LangGraph native checkpoint persistence protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        checkpoint_id = str(checkpoint.get("id") or f"chk_{int(time.time()*1000)}")
        channel_values = checkpoint.get("channel_values", {})
        if not isinstance(channel_values, dict):
            channel_values = {}
        messages = channel_values.get("messages", [])
        intent = str(channel_values.get("intent") or "conversation")
        step_count = int(channel_values.get("step_count") or 0)
        self.save_checkpoint(thread_id=tid, checkpoint_data=messages, intent=intent, step_count=step_count)
        return {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """LangGraph task write staging protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            with self._connect() as conn:
                for channel, value in writes:
                    val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_writes (thread_id, task_id, channel, value_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (tid, str(task_id), str(channel), val_str, now_iso),
                    )
                conn.commit()
        except Exception:
            pass

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """LangGraph native checkpoint restoration protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        data = self.load_checkpoint(tid)
        if not data:
            return None
        checkpoint_id = f"chk_{tid}"
        ckpt = {
            "v": 1,
            "id": checkpoint_id,
            "ts": data.get("updated_at") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "channel_values": {
                "messages": data.get("messages") or [],
                "intent": data.get("intent") or "conversation",
                "step_count": int(data.get("step_count") or 0),
            },
            "channel_versions": {},
            "versions_seen": {},
        }
        meta = {
            "source": "input",
            "step": int(data.get("step_count") or 0),
            "writes": {},
            "parents": {},
        }
        ret_cfg = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }
        return CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ):
        """List checkpoint tuples matching thread_id."""
        tid = None
        if isinstance(config, dict):
            tid = config.get("configurable", {}).get("thread_id")
        with contextlib.closing(self._connect()) as conn:
            if tid:
                rows = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints WHERE thread_id = ? ORDER BY updated_at DESC",
                    (tid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints ORDER BY updated_at DESC",
                ).fetchall()
        count = 0
        for row in rows:
            if limit and count >= limit:
                break
            rtid = row[0]
            r_msgs = json.loads(row[1]) if row[1] else []
            r_intent = row[2] or "conversation"
            r_step = int(row[3] or 0)
            r_upd = row[4]
            chk_id = f"chk_{rtid}"
            ckpt = {
                "v": 1,
                "id": chk_id,
                "ts": r_upd,
                "channel_values": {"messages": r_msgs, "intent": r_intent, "step_count": r_step},
                "channel_versions": {},
                "versions_seen": {},
            }
            meta = {"source": "input", "step": r_step, "writes": {}, "parents": {}}
            ret_cfg = {"configurable": {"thread_id": rtid, "checkpoint_ns": "", "checkpoint_id": chk_id}}
            yield CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)
            count += 1


class InMemorySessionStore(SessionStore, BaseCheckpointSaver):
    """Pure in-memory session store for testing and ephemeral workloads."""

    def __init__(self) -> None:
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._writes: List[Dict[str, Any]] = []

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_data: Union[Dict[str, Any], Sequence[Dict[str, Any]], None] = None,
        intent: str = "conversation",
        step_count: int = 0,
        *,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        tid = (thread_id or "default").strip() or "default"
        if messages is not None and checkpoint_data is None:
            checkpoint_data = messages
        if isinstance(checkpoint_data, dict):
            raw_messages = checkpoint_data.get("messages", [])
            intent = str(checkpoint_data.get("intent", intent))
            step_count = int(checkpoint_data.get("step_count", step_count))
        elif isinstance(checkpoint_data, (list, tuple)):
            raw_messages = checkpoint_data
        else:
            raw_messages = []
        clean_msgs = _clean_messages_for_storage(raw_messages)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._storage[tid] = {
            "thread_id": tid,
            "messages": clean_msgs,
            "intent": intent,
            "step_count": int(step_count),
            "updated_at": now_iso,
        }

    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        tid = (thread_id or "default").strip() or "default"
        data = self._storage.get(tid)
        if not data:
            return None
        return {
            "thread_id": data["thread_id"],
            "messages": [dict(m) for m in data["messages"]],
            "intent": data["intent"],
            "step_count": data["step_count"],
            "updated_at": data["updated_at"],
        }

    def get_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        cp = self.load_checkpoint(thread_id)
        if not cp:
            return []
        return list(cp.get("messages") or [])

    def clear(self, thread_id: str = "default") -> bool:
        tid = (thread_id or "default").strip() or "default"
        if tid in self._storage:
            del self._storage[tid]
            self._writes = [w for w in self._writes if w.get("thread_id") != tid]
            return True
        return False

    def clear_checkpoint(self, thread_id: str) -> None:
        """Backwards compatibility alias for `clear()`."""
        self.clear(thread_id)

    def list_threads(self) -> List[str]:
        return sorted(self._storage.keys())

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LangGraph native checkpoint persistence protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        checkpoint_id = str(checkpoint.get("id") or f"chk_{int(time.time()*1000)}")
        channel_values = checkpoint.get("channel_values", {})
        if not isinstance(channel_values, dict):
            channel_values = {}
        messages = channel_values.get("messages", [])
        intent = str(channel_values.get("intent") or "conversation")
        step_count = int(channel_values.get("step_count") or 0)
        self.save_checkpoint(thread_id=tid, checkpoint_data=messages, intent=intent, step_count=step_count)
        return {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """LangGraph task write staging protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for channel, value in writes:
            self._writes.append({
                "thread_id": tid,
                "task_id": str(task_id),
                "channel": str(channel),
                "value": value,
                "created_at": now_iso,
            })

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """LangGraph native checkpoint restoration protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        data = self.load_checkpoint(tid)
        if not data:
            return None
        checkpoint_id = f"chk_{tid}"
        ckpt = {
            "v": 1,
            "id": checkpoint_id,
            "ts": data.get("updated_at") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "channel_values": {
                "messages": data.get("messages") or [],
                "intent": data.get("intent") or "conversation",
                "step_count": int(data.get("step_count") or 0),
            },
            "channel_versions": {},
            "versions_seen": {},
        }
        meta = {
            "source": "input",
            "step": int(data.get("step_count") or 0),
            "writes": {},
            "parents": {},
        }
        ret_cfg = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }
        return CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ):
        """List checkpoint tuples matching thread_id."""
        tid = None
        if isinstance(config, dict):
            tid = config.get("configurable", {}).get("thread_id")
        items = list(self._storage.values())
        if tid:
            items = [item for item in items if item["thread_id"] == tid]
        items.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        count = 0
        for item in items:
            if limit and count >= limit:
                break
            rtid = item["thread_id"]
            chk_id = f"chk_{rtid}"
            ckpt = {
                "v": 1,
                "id": chk_id,
                "ts": item.get("updated_at"),
                "channel_values": {
                    "messages": item.get("messages") or [],
                    "intent": item.get("intent") or "conversation",
                    "step_count": item.get("step_count") or 0,
                },
                "channel_versions": {},
                "versions_seen": {},
            }
            meta = {"source": "input", "step": item.get("step_count") or 0, "writes": {}, "parents": {}}
            ret_cfg = {"configurable": {"thread_id": rtid, "checkpoint_ns": "", "checkpoint_id": chk_id}}
            yield CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)
            count += 1


def get_session_history(thread_id: str = "default", db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return persisted conversational messages for `thread_id` from `checkpoints.sqlite`."""
    store = SqliteSessionStore(db_path=db_path)
    return store.get_history(thread_id=thread_id)


__all__ = [
    "BaseCheckpointSaver",
    "CheckpointTuple",
    "InMemorySessionStore",
    "SessionStore",
    "SqliteSessionStore",
    "get_session_history",
    "resolve_checkpoint_db_path",
]
