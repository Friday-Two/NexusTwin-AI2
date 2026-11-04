import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH.resolve()))

from memory.store import MemoryStore
from memory.preferences import PreferenceManager
from memory.profile_integration import build_effective_profile

SCHEMA_PATH = PROJECT_ROOT / "memory" / "schema.sql"


class TestPhase10ProfileIntegration(unittest.TestCase):

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

        self.user_id = "integration-test-user"
        self.store.register_user(self.user_id)

        self.profile = {
            "city": "Chennai",
            "preferred_area": "central Chennai",
            "nights": 2,
            "max_total_price": 6000,
            "require_verified": True,
            "privacy": {
                "share_phone": False
            },
            "preferences": {
                "prefer_lower_price": True
            }
        }

    def tearDown(self):
        self.manager = None
        self.store = None

        import gc
        gc.collect()

        self.temp_dir.cleanup()

    def grant_preference_consent(self):
        self.store.set_consent(
            self.user_id,
            "preference_storage",
            True
        )
        self.store.set_consent(
            self.user_id,
            "preference_retrieval",
            True
        )

    def test_missing_user_id_returns_profile_copy(self):
        result = build_effective_profile(
            self.profile,
            user_id=None,
            preference_manager=self.manager
        )

        self.assertEqual(result, self.profile)
        self.assertIsNot(result, self.profile)

    def test_unregistered_user_falls_back(self):
        result = build_effective_profile(
            self.profile,
            user_id="unknown-user",
            preference_manager=self.manager
        )

        self.assertEqual(result, self.profile)

    def test_missing_consent_falls_back(self):
        result = build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertEqual(result, self.profile)

    def test_consented_preferences_are_merged(self):
        self.grant_preference_consent()

        self.manager.set_preference(
            self.user_id,
            "preferred_hotel_type",
            "budget"
        )

        result = build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertEqual(
            result["preferences"]["preferred_hotel_type"],
            "budget"
        )

    def test_budget_cannot_be_overridden(self):
        self.grant_preference_consent()

        self.manager.set_preference(
            self.user_id,
            "max_total_price",
            100000
        )

        result = build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertEqual(result["max_total_price"], 6000)

    def test_verification_cannot_be_overridden(self):
        self.grant_preference_consent()

        self.manager.set_preference(
            self.user_id,
            "require_verified",
            False
        )

        result = build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertTrue(result["require_verified"])

    def test_phone_privacy_cannot_be_overridden(self):
        self.grant_preference_consent()

        self.manager.set_preference(
            self.user_id,
            "share_phone",
            True
        )

        result = build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertFalse(result["privacy"]["share_phone"])

    def test_original_profile_is_not_modified(self):
        self.grant_preference_consent()

        self.manager.set_preference(
            self.user_id,
            "preferred_hotel_type",
            "budget"
        )

        original = json.loads(json.dumps(self.profile))

        build_effective_profile(
            self.profile,
            user_id=self.user_id,
            preference_manager=self.manager
        )

        self.assertEqual(self.profile, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)