import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from execution_verifier import (
    execute_agreement,
    initialize_database,
    verify_persisted_booking,
)


class ExecutionVerifierTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_services.db"

        self.profile = {
            "city": "Chennai",
            "preferred_area": "central Chennai",
            "nights": 2,
            "max_total_price": 6000,
            "require_verified": True,
            "privacy": {"share_phone": False},
        }

        self.agreement = {
            "agent": "Agent_Test",
            "hotel": "Central Inn",
            "area": "central Chennai",
            "total_price": 5500,
            "verified": True,
        }

        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_initialization(self):
        self.assertTrue(self.db_path.exists())

    def test_valid_agreement_is_persisted(self):
        result = execute_agreement(
            self.agreement,
            self.profile,
            self.db_path,
            booking_id="test-booking-001",
        )

        self.assertEqual(result["status"], "persisted")
        self.assertEqual(result["hotel"], "Central Inn")

    def test_duplicate_booking_id_does_not_overwrite_original(self):
        original_agreement = self.agreement.copy()

        execute_agreement(
            original_agreement,
            self.profile,
            self.db_path,
            booking_id="duplicate-test-001",
        )

        changed_agreement = self.agreement.copy()
        changed_agreement["hotel"] = "Different Hotel"
        changed_agreement["total_price"] = 5800

        with self.assertRaises(sqlite3.IntegrityError):
            execute_agreement(
                changed_agreement,
                self.profile,
                self.db_path,
                booking_id="duplicate-test-001",
            )

        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT hotel, total_price FROM bookings WHERE booking_id = ?",
                ("duplicate-test-001",),
            ).fetchone()

        self.assertEqual(row[0], original_agreement["hotel"])
        self.assertEqual(row[1], original_agreement["total_price"])

    def test_persisted_agreement_passes_verification(self):
        booking_id = "test-booking-002"

        execute_agreement(
            self.agreement,
            self.profile,
            self.db_path,
            booking_id=booking_id,
        )

        result = verify_persisted_booking(
            booking_id,
            self.agreement,
            self.profile,
            self.db_path,
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["errors"], [])

    def test_invalid_agreement_is_not_persisted(self):
        invalid = dict(self.agreement)
        invalid["total_price"] = 9000

        with self.assertRaises(ValueError):
            execute_agreement(
                invalid,
                self.profile,
                self.db_path,
                booking_id="invalid-booking",
            )

        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM bookings WHERE booking_id = ?",
                ("invalid-booking",),
            ).fetchone()

        self.assertIsNone(row)

    def test_missing_booking_fails_verification(self):
        result = verify_persisted_booking(
            "missing-booking",
            self.agreement,
            self.profile,
            self.db_path,
        )

        self.assertFalse(result["verified"])
        self.assertIn("booking_record_missing", result["errors"])

    def test_tampered_price_is_detected(self):
        booking_id = "test-booking-003"

        execute_agreement(
            self.agreement,
            self.profile,
            self.db_path,
            booking_id=booking_id,
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE bookings SET total_price = ? WHERE booking_id = ?",
                (5900, booking_id),
            )
            conn.commit()

        result = verify_persisted_booking(
            booking_id,
            self.agreement,
            self.profile,
            self.db_path,
        )

        self.assertFalse(result["verified"])
        self.assertTrue(result["errors"])

    def test_tampered_agreement_snapshot_is_detected(self):
        booking_id = "test-booking-004"

        execute_agreement(
            self.agreement,
            self.profile,
            self.db_path,
            booking_id=booking_id,
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE bookings SET agreement_json = ? WHERE booking_id = ?",
                (json.dumps({"hotel": "Fake Hotel"}), booking_id),
            )
            conn.commit()

        result = verify_persisted_booking(
            booking_id,
            self.agreement,
            self.profile,
            self.db_path,
        )

        self.assertFalse(result["verified"])
        self.assertTrue(
            any("agreement_snapshot" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)