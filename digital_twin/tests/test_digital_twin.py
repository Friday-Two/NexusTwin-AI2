import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import digital_twin as digital_twin_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from digital_twin import (
    check_constraints,
    negotiate,
    parse_request,
    sanitize_outgoing_message,
    validate_final_agreement,
)

PROFILE_PATH = PROJECT_ROOT / "profiles" / "digital_twin_profile.json"


class DigitalTwinTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_profile = json.loads(
            PROFILE_PATH.read_text(encoding="utf-8")
        )

    def setUp(self):
        self.profile = copy.deepcopy(self.base_profile)

    # 1. Request parsing

    def test_parse_request_extracts_nights_and_budget(self):
        request = "Book a hotel for 2 nights under Rs. 5000"

        result = parse_request(request, self.profile)

        self.assertEqual(result["nights"], 2)
        self.assertEqual(result["max_total_price"], 5000)

    def test_request_cannot_increase_profile_budget(self):
        request = "Book a hotel for 2 nights under Rs. 9000"

        result = parse_request(request, self.profile)

        self.assertEqual(result["max_total_price"], 6000)

    def test_profile_privacy_denial_is_preserved(self):
        request = "Book a hotel. Share my phone."

        result = parse_request(request, self.profile)

        self.assertFalse(result["privacy"]["share_phone"])

    # 2. Hard constraints

    def test_valid_offer_passes_constraints(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Central Inn",
            "area": "central Chennai",
            "total_price": 5500,
            "verified": True,
        }

        valid, reasons = check_constraints(offer, self.profile)

        self.assertTrue(valid)
        self.assertEqual(reasons, [])

    def test_over_budget_offer_is_rejected(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Central Inn",
            "area": "central Chennai",
            "total_price": 7000,
            "verified": True,
        }

        valid, reasons = check_constraints(offer, self.profile)

        self.assertFalse(valid)
        self.assertTrue(any("price" in reason for reason in reasons))

    def test_unverified_hotel_is_rejected(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Central Inn",
            "area": "central Chennai",
            "total_price": 5500,
            "verified": False,
        }

        valid, reasons = check_constraints(offer, self.profile)

        self.assertFalse(valid)
        self.assertIn("hotel_not_verified", reasons)

    def test_wrong_area_is_rejected(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Other Hotel",
            "area": "North Chennai",
            "total_price": 5500,
            "verified": True,
        }

        valid, reasons = check_constraints(offer, self.profile)

        self.assertFalse(valid)
        self.assertIn("area_mismatch", reasons)

    # 3. Privacy protection

    def test_private_fields_are_removed_recursively(self):
        message = {
            "hotel": "Central Inn",
            "phone": "9876543210",
            "details": {
                "mobile_number": "9876543210",
                "agent": "Agent_Test",
            },
        }

        result = sanitize_outgoing_message(message, self.profile)

        self.assertNotIn("phone", result)
        self.assertNotIn("mobile_number", result["details"])
        self.assertEqual(result["details"]["agent"], "Agent_Test")

    def test_phone_number_in_text_is_redacted(self):
        message = {"note": "Call me at 9876543210"}

        result = sanitize_outgoing_message(message, self.profile)

        self.assertIn("[REDACTED_PHONE]", result["note"])
        self.assertNotIn("9876543210", result["note"])

    def test_phone_is_preserved_when_sharing_is_allowed(self):
        self.profile["privacy"]["share_phone"] = True
        message = {"phone": "9876543210"}

        result = sanitize_outgoing_message(message, self.profile)

        self.assertEqual(result["phone"], "9876543210")

    def test_nested_phone_number_is_redacted(self):
        message = {
            "agent": "Agent_Test",
            "details": {
                "contact_info": {
                    "note": "Customer number: +91 98765 43210"
                }
            },
        }

        result = sanitize_outgoing_message(message, self.profile)

        note = result["details"]["contact_info"]["note"]

        self.assertIn("[REDACTED_PHONE]", note)
        self.assertNotIn("98765 43210", note)

    def test_alternative_phone_field_names_are_removed(self):
        message = {
            "phone": "9876543210",
            "mobile": "9876543210",
            "telephone": "9876543210",
            "contact_number": "9876543210",
            "agent": "Agent_Test",
        }

        result = sanitize_outgoing_message(message, self.profile)

        self.assertNotIn("phone", result)
        self.assertNotIn("mobile", result)
        self.assertNotIn("telephone", result)
        self.assertNotIn("contact_number", result)
        self.assertEqual(result["agent"], "Agent_Test")

    def test_phone_sharing_policy_does_not_modify_original_message(self):
        message = {
            "phone": "9876543210",
            "details": {
                "note": "Call 9876543210"
            },
        }

        original = {
            "phone": "9876543210",
            "details": {
                "note": "Call 9876543210"
            },
        }

        sanitize_outgoing_message(message, self.profile)

        self.assertEqual(message, original)

    def test_privacy_audit_detects_phone_leak(self):
        # Simulate a sanitizer failure so the audit receives a leaked number.
        def inject_phone(offer, profile):
            result = offer.copy()
            result["phone"] = "9876543210"
            return result

        with patch.object(
            digital_twin_module,
            "public_offer",
            side_effect=inject_phone,
        ), patch.object(
            digital_twin_module,
            "sanitize_outgoing_message",
            side_effect=lambda message, profile: message,
        ):
            trace = negotiate(
                "Book a verified hotel in central Chennai "
                "for 2 nights under Rs. 6000.",
                self.profile,
                max_rounds=3,
            )

        self.assertTrue(
            trace["privacy_audit"]["phone_disclosed_to_agents"]
        )
        self.assertTrue(
            trace["privacy_audit"]["private_fields_in_outgoing_offers"]
        )

    # 4. Negotiation

    def test_normal_request_reaches_agreement(self):
        request = (
            "Book a verified hotel in central Chennai "
            "for 2 nights under Rs. 6000. "
            "Do NOT share my phone."
        )

        trace = negotiate(request, self.profile, max_rounds=3)

        self.assertEqual(trace["status"], "agreement")
        self.assertIsNotNone(trace["accepted_offer"])
        self.assertLessEqual(
            trace["accepted_offer"]["total_price"], 6000
        )

    def test_impossible_budget_does_not_reach_agreement(self):
        request = "Book a verified hotel for 2 nights under Rs. 1000"

        trace = negotiate(request, self.profile, max_rounds=3)

        self.assertEqual(trace["status"], "failed")
        self.assertIsNone(trace["accepted_offer"])

    def test_zero_negotiation_rounds_are_rejected(self):
        with self.assertRaises(ValueError):
            negotiate("Book a hotel", self.profile, max_rounds=0)

    # 5. Final agreement validation

    def test_valid_final_agreement_passes(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Central Inn",
            "area": "central Chennai",
            "total_price": 5500,
            "verified": True,
        }

        valid, reasons = validate_final_agreement(
            offer, self.profile
        )

        self.assertTrue(valid)
        self.assertEqual(reasons, [])

    def test_invalid_final_agreement_fails(self):
        offer = {
            "agent": "Agent_Test",
            "hotel": "Unverified Hotel",
            "area": "central Chennai",
            "total_price": 7000,
            "verified": False,
        }

        valid, reasons = validate_final_agreement(
            offer, self.profile
        )

        self.assertFalse(valid)
        self.assertIn("hotel_not_verified", reasons)

    def test_non_dictionary_agreement_is_rejected(self):
        valid, reasons = validate_final_agreement(
            None, self.profile
        )

        self.assertFalse(valid)
        self.assertIn("agreement_is_not_an_object", reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)