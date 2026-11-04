import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = (
    PROJECT_ROOT / "memory" / "digital_twin_memory.db"
)

MEMORY_POLICY_VERSION = "1.0"


class ConsentRequiredError(PermissionError):
    """Raised when a memory operation lacks explicit user consent."""


class _ClosingConnection:
    """SQLite connection context manager that also closes the connection."""
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.__enter__()
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.connection.__exit__(
                exc_type, exc_value, traceback
            )
        finally:
            self.connection.close()


class MemoryStore:
    """Persistent, consent-controlled digital twin memory storage."""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)

        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"Memory database does not exist: {self.db_path}"
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return _ClosingConnection(conn)

    def register_user(self, user_id):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        user_id = user_id.strip()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_users (user_id)
                VALUES (?)
                """,
                (user_id,)
            )

            self._audit(
                conn,
                user_id,
                "register",
                "user",
                user_id
            )

        return user_id

    def set_consent(self, user_id, scope, granted):
        self._validate_user_id(user_id)

        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("Consent scope must be non-empty")

        if not isinstance(granted, bool):
            raise ValueError("granted must be a boolean")

        scope = scope.strip()
        consent_id = str(uuid.uuid4())

        with self._connect() as conn:
            self._require_user(conn, user_id)

            conn.execute(
                """
                INSERT INTO consents (
                    consent_id,
                    user_id,
                    consent_scope,
                    granted,
                    policy_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, consent_scope)
                DO UPDATE SET
                    granted = excluded.granted,
                    policy_version = excluded.policy_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    consent_id,
                    user_id,
                    scope,
                    int(granted),
                    MEMORY_POLICY_VERSION
                )
            )

            self._audit(
                conn,
                user_id,
                "consent_granted" if granted else "consent_revoked",
                "consent",
                scope,
                {"granted": granted}
            )

        return granted

    def has_consent(self, user_id, scope):
        self._validate_user_id(user_id)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT granted
                FROM consents
                WHERE user_id = ? AND consent_scope = ?
                """,
                (user_id, scope)
            ).fetchone()

        return bool(row["granted"]) if row else False

    def store_memory(
        self,
        user_id,
        category,
        content,
        source="user",
        sensitivity="normal",
        expires_at=None
    ):
        self._validate_user_id(user_id)

        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be non-empty")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be non-empty")

        if sensitivity != "normal":
            raise ValueError(
                "Only normal-sensitivity memories are supported"
            )

        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be non-empty")

        memory_id = str(uuid.uuid4())

        with self._connect() as conn:
            self._require_user(conn, user_id)
            self._require_consent(
                conn,
                user_id,
                "memory_storage"
            )

            conn.execute(
                """
                INSERT INTO memories (
                    memory_id,
                    user_id,
                    category,
                    content,
                    source,
                    sensitivity,
                    status,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, 'normal', 'active', ?)
                """,
                (
                    memory_id,
                    user_id,
                    category.strip(),
                    content.strip(),
                    source.strip(),
                    expires_at
                )
            )

            self._audit(
                conn,
                user_id,
                "memory_created",
                "memory",
                memory_id,
                {"category": category.strip()}
            )

        return memory_id

    def retrieve_memories(self, user_id, category=None):
        self._validate_user_id(user_id)

        with self._connect() as conn:
            self._require_user(conn, user_id)
            self._require_consent(
                conn,
                user_id,
                "memory_retrieval"
            )

            query = """
                SELECT memory_id, category, content, source,
                       created_at, expires_at
                FROM memories
                WHERE user_id = ?
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            params = [user_id]

            if category is not None:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY created_at DESC"

            rows = conn.execute(query, params).fetchall()

            self._audit(
                conn,
                user_id,
                "memory_retrieved",
                "memory",
                None,
                {"count": len(rows)}
            )

        return [dict(row) for row in rows]

    def _validate_user_id(self, user_id):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

    def _require_user(self, conn, user_id):
        row = conn.execute(
            "SELECT 1 FROM memory_users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if row is None:
            raise ValueError("User is not registered")

    def _require_consent(self, conn, user_id, scope):
        row = conn.execute(
            """
            SELECT granted
            FROM consents
            WHERE user_id = ? AND consent_scope = ?
            """,
            (user_id, scope)
        ).fetchone()

        if row is None or row["granted"] != 1:
            raise ConsentRequiredError(
                f"Explicit consent required for: {scope}"
            )

    def _audit(
        self,
        conn,
        user_id,
        operation,
        entity_type,
        entity_id=None,
        details=None
    ):
        conn.execute(
            """
            INSERT INTO memory_audit (
                audit_id,
                user_id,
                operation,
                entity_type,
                entity_id,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                operation,
                entity_type,
                entity_id,
                json.dumps(details) if details is not None else None
            )
        )

    def delete_memory(self, user_id, memory_id):
        """Permanently delete one memory owned by the user."""

        self._validate_user_id(user_id)

        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be non-empty")

        with self._connect() as conn:
            self._require_user(conn, user_id)

            row = conn.execute(
                """
                SELECT memory_id
                FROM memories
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id)
            ).fetchone()

            if row is None:
                return False

            conn.execute(
                """
                DELETE FROM memories
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id)
            )

            self._audit(
                conn,
                user_id,
                "memory_deleted",
                "memory",
                memory_id
            )

        return True