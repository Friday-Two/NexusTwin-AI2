import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ai.allocation_validator import validate_allocation


class AllocationValidatorTests(unittest.TestCase):

    OWN_INPUT = "1 4 4 1 1 2"

    def test_valid_allocation(self):
        target = (
            "item0=1 item1=0 item2=1 "
            "item0=0 item1=4 item2=0"
        )
        result = validate_allocation(target, self.OWN_INPUT)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.reason, "valid")
        self.assertEqual(result.allocated_totals, (1, 4, 1))

    def test_allocation_that_does_not_conserve_items(self):
        target = (
            "item0=1 item1=0 item2=1 "
            "item0=1 item1=0 item2=1"
        )
        result = validate_allocation(target, self.OWN_INPUT)

        self.assertFalse(result.is_valid)
        self.assertEqual(
            result.reason,
            "allocation_does_not_conserve_items",
        )

    def test_missing_assignment_is_rejected(self):
        result = validate_allocation(
            "item0=1 item1=0 item2=1",
            self.OWN_INPUT,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "expected_six_assignments")

    def test_wrong_assignment_order_is_rejected(self):
        target = (
            "item1=1 item0=0 item2=1 "
            "item0=0 item1=4 item2=0"
        )
        result = validate_allocation(target, self.OWN_INPUT)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "unexpected_assignment_order")

    def test_invalid_input_length_is_rejected(self):
        target = (
            "item0=1 item1=0 item2=1 "
            "item0=0 item1=4 item2=0"
        )
        result = validate_allocation(target, "1 4 4")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "invalid_input_length")

    def test_negative_input_value_is_rejected(self):
        target = (
            "item0=1 item1=0 item2=1 "
            "item0=0 item1=4 item2=0"
        )
        result = validate_allocation(target, "1 -4 4 1 1 2")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "invalid_input_values")

    def test_negative_allocation_value_is_rejected(self):
        target = (
            "item0=-1 item1=0 item2=1 "
            "item0=2 item1=4 item2=0"
        )
        result = validate_allocation(target, self.OWN_INPUT)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "negative_allocation_value")
        self.assertEqual(result.assignments[0], ("item0", -1))

    def test_non_string_target_is_rejected(self):
        with self.assertRaises(TypeError):
            validate_allocation(None, self.OWN_INPUT)

    def test_non_string_input_is_rejected(self):
        with self.assertRaises(TypeError):
            validate_allocation("item0=1", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)