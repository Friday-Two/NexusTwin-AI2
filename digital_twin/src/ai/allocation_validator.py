import re
from dataclasses import dataclass
from typing import Tuple


EXPECTED_ASSIGNMENT_NAMES = (
    "item0", "item1", "item2",
    "item0", "item1", "item2",
)


@dataclass(frozen=True)
class AllocationValidation:
    is_valid: bool
    reason: str
    assignments: Tuple[Tuple[str, int], ...]
    allocated_totals: Tuple[int, ...]


def validate_allocation(
    target: str,
    own_input: str,
) -> AllocationValidation:
    """
    Validate a six-assignment allocation using the original
    TravelPlanner/Deal-or-No-Deal dataset validation rules.
    """

    if not isinstance(target, str):
        raise TypeError("Allocation target must be a string.")

    if not isinstance(own_input, str):
        raise TypeError("Own input must be a string.")

    pairs = tuple(
        (name, int(value))
        for name, value in re.findall(
            r"\b(item[0-9]+)\s*=\s*(-?\d+)",
            target,
        )
    )

    def reject(reason: str) -> AllocationValidation:
        return AllocationValidation(
            is_valid=False,
            reason=reason,
            assignments=pairs,
            allocated_totals=(),
        )

    if len(pairs) != 6:
        return reject("expected_six_assignments")

    if any(value < 0 for _, value in pairs):
        return reject("negative_allocation_value")

    names = tuple(name for name, _ in pairs)

    if names != EXPECTED_ASSIGNMENT_NAMES:
        return reject("unexpected_assignment_order")

    numbers_text = own_input.split()

    if len(numbers_text) != 6:
        return reject("invalid_input_length")

    if any(not re.fullmatch(r"\d+", value) for value in numbers_text):
        return reject("invalid_input_values")

    numbers = [int(value) for value in numbers_text]
    values = [value for _, value in pairs]

    counts = (
        numbers[0],
        numbers[2],
        numbers[4],
    )

    allocated_totals = (
        values[0] + values[3],
        values[1] + values[4],
        values[2] + values[5],
    )

    if allocated_totals != counts:
        return AllocationValidation(
            is_valid=False,
            reason="allocation_does_not_conserve_items",
            assignments=pairs,
            allocated_totals=allocated_totals,
        )

    return AllocationValidation(
        is_valid=True,
        reason="valid",
        assignments=pairs,
        allocated_totals=allocated_totals,
    )