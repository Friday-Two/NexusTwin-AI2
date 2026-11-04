
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH.resolve()))

from memory.store import MemoryStore, ConsentRequiredError

SCHEMA_PATH = PROJECT_ROOT / "memory" / "schema.sql"


class TestPhase10Memory(unittest.TestCase):

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
        self.user_id = "memory-test-user"

        self.store.register_user(self.user_id)

    def tearDown(self):
        self.store = None

        import gc
        gc.collect()

        self.temp_dir.cleanup()

    def grant_storage(self, user_id=None):
        self.store.set_consent(
            user_id or self.user_id,
            "memory_storage",
            True
        )

    def grant_retrieval(self, user_id=None):
        self.store.set_consent(
            user_id or self.user_id,
            "memory_retrieval",
            True
        )

    def test_storage_requires_consent(self):
        with self.assertRaises(ConsentRequiredError):
            self.store.store_memory(
                self.user_id,
                "travel",
                "Prefers verified hotels"
            )

    def test_retrieval_requires_consent(self):
        self.grant_storage()

        self.store.store_memory(
            self.user_id,
            "travel",
            "Prefers verified hotels"
        )

        with self.assertRaises(ConsentRequiredError):
            self.store.retrieve_memories(self.user_id)

    def test_store_and_retrieve_memory(self):
        self.grant_storage()
        self.grant_retrieval()

        memory_id = self.store.store_memory(
            self.user_id,
            "travel",
            "Prefers verified hotels",
            source="explicit_user_input"
        )

        memories = self.store.retrieve_memories(self.user_id)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["memory_id"], memory_id)
        self.assertEqual(
            memories[0]["content"],
            "Prefers verified hotels"
        )
        self.assertEqual(memories[0]["category"], "travel")

    def test_users_are_isolated(self):
        self.grant_storage()

        self.store.store_memory(
            self.user_id,
            "travel",
            "User one preference"
        )

        second_user = "memory-test-user-2"
        self.store.register_user(second_user)

        self.grant_retrieval(second_user)

        memories = self.store.retrieve_memories(second_user)

        self.assertEqual(memories, [])

    def test_revoking_storage_consent_blocks_storage(self):
        self.grant_storage()

        self.store.set_consent(
            self.user_id,
            "memory_storage",
            False
        )

        with self.assertRaises(ConsentRequiredError):
            self.store.store_memory(
                self.user_id,
                "travel",
                "New preference"
            )

    def test_revoking_retrieval_consent_blocks_retrieval(self):
        self.grant_storage()
        self.grant_retrieval()

        self.store.store_memory(
            self.user_id,
            "travel",
            "Prefers verified hotels"
        )

        self.store.set_consent(
            self.user_id,
            "memory_retrieval",
            False
        )

        with self.assertRaises(ConsentRequiredError):
            self.store.retrieve_memories(self.user_id)

    def test_delete_without_retrieval_consent(self):
        self.grant_storage()

        memory_id = self.store.store_memory(
            self.user_id,
            "travel",
            "Temporary preference"
        )

        deleted = self.store.delete_memory(
            self.user_id,
            memory_id
        )

        self.assertTrue(deleted)

    def test_delete_nonexistent_memory(self):
        result = self.store.delete_memory(
            self.user_id,
            "nonexistent-memory-id"
        )

        self.assertFalse(result)

    def test_sensitive_memory_is_rejected(self):
        self.grant_storage()

        with self.assertRaises(ValueError):
            self.store.store_memory(
                self.user_id,
                "personal",
                "Sensitive information",
                sensitivity="sensitive"
            )

    def test_memory_deletion_is_audited(self):
        self.grant_storage()

        memory_id = self.store.store_memory(
            self.user_id,
            "travel",
            "Temporary preference"
        )

        self.store.delete_memory(
            self.user_id,
            memory_id
        )

        with self.store._connect() as conn:
            audit = conn.execute(
                """
                SELECT operation, entity_type, entity_id,
                       details_json
                FROM memory_audit
                WHERE user_id = ?
                  AND operation = 'memory_deleted'
                """,
                (self.user_id,)
            ).fetchone()

        self.assertIsNotNone(audit)
        self.assertEqual(audit["operation"], "memory_deleted")
        self.assertEqual(audit["entity_type"], "memory")
        self.assertEqual(audit["entity_id"], memory_id)
        self.assertIsNone(audit["details_json"])

    def test_category_filter(self):
        self.grant_storage()
        self.grant_retrieval()

        self.store.store_memory(
            self.user_id,
            "travel",
            "Prefers verified hotels"
        )

        self.store.store_memory(
            self.user_id,
            "food",
            "Prefers vegetarian restaurants"
        )

        memories = self.store.retrieve_memories(
            self.user_id,
            category="travel"
        )

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["category"], "travel")

    def test_invalid_empty_memory_content(self):
        self.grant_storage()

        with self.assertRaises(ValueError):
            self.store.store_memory(
                self.user_id,
                "travel",
                "   "
            )


if __name__ == "__main__":
    unittest.main()
