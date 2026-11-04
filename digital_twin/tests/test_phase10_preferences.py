import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH.resolve()))

from memory.store import MemoryStore, ConsentRequiredError
from memory.preferences import PreferenceManager

SCHEMA_PATH = PROJECT_ROOT / "memory" / "schema.sql"


class TestPhase10Preferences(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                SCHEMA_PATH.read_text(encoding="utf-8")
            )
            conn.commit()
        finally:
            conn.close()

        self.store = MemoryStore(self.db_path)
        self.manager = PreferenceManager(self.store)

        self.user_id = "test-user-1"
        self.store.register_user(self.user_id)

    def tearDown(self):
        self.manager = None
        self.store = None

        import gc
        gc.collect()

        self.temp_dir.cleanup()

    def grant_storage(self):
        self.store.set_consent(
            self.user_id, "preference_storage", True
        )

    def grant_retrieval(self):
        self.store.set_consent(
            self.user_id, "preference_retrieval", True
        )

    def test_storage_requires_consent(self):
        with self.assertRaises(ConsentRequiredError):
            self.manager.set_preference(
                self.user_id, "budget", 5000
            )

    def test_retrieval_requires_consent(self):
        self.grant_storage()
        self.manager.set_preference(
            self.user_id, "budget", 5000
        )

        with self.assertRaises(ConsentRequiredError):
            self.manager.get_preferences(self.user_id)

    def test_save_and_retrieve_structured_preference(self):
        self.grant_storage()
        self.grant_retrieval()

        value = {
            "max_price": 5000,
            "currency": "INR",
            "verified_only": True
        }

        self.manager.set_preference(
            self.user_id, "hotel_budget", value
        )

        result = self.manager.get_preferences(self.user_id)

        self.assertEqual(result["hotel_budget"], value)

    def test_update_does_not_create_duplicate(self):
        self.grant_storage()

        self.manager.set_preference(
            self.user_id, "budget", 5000
        )
        self.manager.set_preference(
            self.user_id, "budget", 4500
        )

        with self.store._connect() as conn:
            count = conn.execute(
                """
                SELECT COUNT(*) FROM preferences
                WHERE user_id = ? AND preference_key = ?
                """,
                (self.user_id, "budget")
            ).fetchone()[0]

        self.assertEqual(count, 1)

    def test_users_are_isolated(self):
        self.grant_storage()
        self.manager.set_preference(
            self.user_id, "budget", 5000
        )

        self.store.register_user("test-user-2")
        self.store.set_consent(
            "test-user-2", "preference_storage", True
        )
        self.store.set_consent(
            "test-user-2", "preference_retrieval", True
        )

        self.assertEqual(
            self.manager.get_preferences("test-user-2"), {}
        )

    def test_delete_without_retrieval_consent(self):
        self.grant_storage()
        self.manager.set_preference(
            self.user_id, "budget", 5000
        )

        deleted = self.manager.delete_preference(
            self.user_id, "budget"
        )

        self.assertTrue(deleted)

        with self.store._connect() as conn:
            count = conn.execute(
                """
                SELECT COUNT(*) FROM preferences
                WHERE user_id = ? AND preference_key = ?
                """,
                (self.user_id, "budget")
            ).fetchone()[0]

        self.assertEqual(count, 0)

    def test_delete_nonexistent_preference(self):
        result = self.manager.delete_preference(
            self.user_id, "does_not_exist"
        )
        self.assertFalse(result)

    def test_invalid_preference_key(self):
        self.grant_storage()

        with self.assertRaises(ValueError):
            self.manager.set_preference(
                self.user_id, "   ", 5000
            )

    def test_none_value_is_rejected(self):
        self.grant_storage()

        with self.assertRaises(ValueError):
            self.manager.set_preference(
                self.user_id, "budget", None
            )

    def test_preference_deletion_is_audited(self):
        self.grant_storage()
        self.manager.set_preference(
            self.user_id, "budget", 5000
        )
        self.manager.delete_preference(
            self.user_id, "budget"
        )

        with self.store._connect() as conn:
            audit = conn.execute(
                """
                SELECT operation, entity_type, entity_id
                FROM memory_audit
                WHERE user_id = ?
                  AND operation = 'preference_deleted'
                """,
                (self.user_id,)
            ).fetchone()

        self.assertIsNotNone(audit)
        self.assertEqual(audit["operation"], "preference_deleted")
        self.assertEqual(audit["entity_type"], "preference")
        self.assertEqual(audit["entity_id"], "budget")


if __name__ == "__main__":
    unittest.main()
