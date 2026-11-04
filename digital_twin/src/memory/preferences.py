import json
import uuid

from .store import MemoryStore, ConsentRequiredError


class PreferenceManager:
    """Manage persistent, consent-controlled user preferences."""

    def __init__(self, store=None):
        self.store = store or MemoryStore()

    def set_preference(self, user_id, key, value):
        self.store._validate_user_id(user_id)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Preference key must be non-empty")

        if value is None:
            raise ValueError("Preference value cannot be None")

        key = key.strip()
        serialized_value = json.dumps(value, ensure_ascii=False)
        preference_id = str(uuid.uuid4())

        with self.store._connect() as conn:
            self.store._require_user(conn, user_id)
            self.store._require_consent(
                conn, user_id, "preference_storage"
            )

            conn.execute(
                """
                INSERT INTO preferences (
                    preference_id, user_id, preference_key,
                    preference_value, source
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, preference_key)
                DO UPDATE SET
                    preference_value = excluded.preference_value,
                    source = excluded.source,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    preference_id,
                    user_id,
                    key,
                    serialized_value,
                    "explicit_user_input"
                )
            )

            self.store._audit(
                conn, user_id, "preference_saved",
                "preference", key
            )

        return key

    def get_preferences(self, user_id):
        self.store._validate_user_id(user_id)

        with self.store._connect() as conn:
            self.store._require_user(conn, user_id)
            self.store._require_consent(
                conn, user_id, "preference_retrieval"
            )

            rows = conn.execute(
                """
                SELECT preference_key, preference_value
                FROM preferences
                WHERE user_id = ?
                ORDER BY preference_key
                """,
                (user_id,)
            ).fetchall()

            self.store._audit(
                conn, user_id, "preferences_retrieved",
                "preference", None,
                {"count": len(rows)}
            )

        return {
            row["preference_key"]: json.loads(row["preference_value"])
            for row in rows
        }

    def delete_preference(self, user_id, key):
        self.store._validate_user_id(user_id)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Preference key must be non-empty")

        key = key.strip()

        with self.store._connect() as conn:
            self.store._require_user(conn, user_id)

            result = conn.execute(
                """
                DELETE FROM preferences
                WHERE user_id = ? AND preference_key = ?
                """,
                (user_id, key)
            )

            if result.rowcount == 0:
                return False

            self.store._audit(
                conn, user_id, "preference_deleted",
                "preference", key
            )

        return True
