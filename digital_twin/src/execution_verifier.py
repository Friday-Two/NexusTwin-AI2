import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    hotel TEXT NOT NULL,
    area TEXT NOT NULL,
    total_price INTEGER NOT NULL,
    nights INTEGER NOT NULL,
    verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
    agent TEXT NOT NULL,
    agreement_json TEXT NOT NULL,
    status TEXT NOT NULL
)
"""


def initialize_database(db_path):
    """Create the simulated execution database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()

    return str(db_path)


def execute_agreement(agreement, profile, db_path, booking_id="booking-001"):
    """
    Persist a validated simulated agreement.
    Recheck hard constraints before writing.
    """
    from digital_twin import validate_final_agreement

    valid, reasons = validate_final_agreement(agreement, profile)
    if not valid:
        raise ValueError(
            f"Refusing to execute invalid agreement: {reasons}"
        )

    nights = int(profile["nights"])
    price = int(agreement["total_price"])

    if nights <= 0:
        raise ValueError("Booking duration must be positive.")

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bookings (
                booking_id, hotel, area, total_price, nights,
                verified, agent, agreement_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                agreement["hotel"],
                agreement["area"],
                price,
                nights,
                int(agreement["verified"] is True),
                agreement["agent"],
                json.dumps(agreement, sort_keys=True),
                "accepted"
            )
        )
        conn.commit()

    return {
        "booking_id": booking_id,
        "database_path": str(db_path),
        "status": "persisted",
        "hotel": agreement["hotel"],
        "total_price": price,
        "nights": nights
    }


def verify_persisted_booking(booking_id, expected_agreement,
                             profile, db_path):
    """
    Independently reopen SQLite and compare persisted values with
    the original agreement and user constraints.
    """
    errors = []
    db_path = Path(db_path)

    if not db_path.exists():
        return {
            "verified": False,
            "booking_id": booking_id,
            "errors": ["database_missing"]
        }

    # Open a fresh connection and read the persisted row.
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?",
            (booking_id,)
        ).fetchone()

    if row is None:
        return {
            "verified": False,
            "booking_id": booking_id,
            "errors": ["booking_record_missing"]
        }

    expected_fields = {
        "hotel": expected_agreement.get("hotel"),
        "area": expected_agreement.get("area"),
        "total_price": expected_agreement.get("total_price"),
        "nights": profile.get("nights"),
        "verified": 1 if expected_agreement.get("verified") is True else 0,
        "agent": expected_agreement.get("agent"),
        "status": "accepted",
    }

    for field, expected in expected_fields.items():
        if row[field] != expected:
            errors.append(
                f"{field}_mismatch: expected={expected!r}, "
                f"stored={row[field]!r}"
            )

    # Verify the stored agreement snapshot agrees with stored columns.
    try:
        snapshot = json.loads(row["agreement_json"])
        for field in ("hotel", "area", "total_price", "verified", "agent"):
            stored_value = row[field]
            if field == "verified":
                stored_value = bool(stored_value)
            if snapshot.get(field) != stored_value:
                errors.append(f"agreement_snapshot_{field}_mismatch")
    except (TypeError, json.JSONDecodeError):
        errors.append("agreement_snapshot_invalid")

    # Independently check user hard constraints against persisted values.
    try:
        if row["total_price"] > int(profile["max_total_price"]):
            errors.append("persisted_price_exceeds_budget")
        if row["nights"] != int(profile["nights"]):
            errors.append("persisted_duration_mismatch")
        if profile.get("require_verified", True) and row["verified"] != 1:
            errors.append("persisted_hotel_not_verified")
        if row["area"].casefold() != str(
            profile["preferred_area"]
        ).casefold():
            errors.append("persisted_area_mismatch")
    except (TypeError, ValueError, KeyError):
        errors.append("profile_or_record_constraint_check_failed")

    return {
        "verified": len(errors) == 0,
        "booking_id": booking_id,
        "hotel": row["hotel"],
        "total_price": row["total_price"],
        "nights": row["nights"],
        "verified_hotel": bool(row["verified"]),
        "errors": errors
    }
