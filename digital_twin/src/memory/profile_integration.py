import copy

from .store import MemoryStore, ConsentRequiredError
from .preferences import PreferenceManager


ALLOWED_SOFT_PREFERENCES = {
    "prefer_lower_price",
    "preferred_amenities",
    "preferred_hotel_type",
}


def build_effective_profile(
    base_profile,
    user_id=None,
    preference_manager=None
):
    """
    Build a profile using consented persistent preferences.

    Hard constraints and privacy settings always remain authoritative
    in the original profile.
    """

    if not isinstance(base_profile, dict):
        raise TypeError("base_profile must be a dictionary")

    effective_profile = copy.deepcopy(base_profile)

    if not isinstance(user_id, str) or not user_id.strip():
        return effective_profile

    manager = preference_manager or PreferenceManager()

    try:
        stored_preferences = manager.get_preferences(user_id.strip())

    except (ConsentRequiredError, ValueError):
        # No consent or unregistered user:
        # continue with the original profile.
        return effective_profile

    if not isinstance(stored_preferences, dict):
        return effective_profile

    profile_preferences = effective_profile.setdefault(
        "preferences", {}
    )

    for key in ALLOWED_SOFT_PREFERENCES:
        if key in stored_preferences:
            profile_preferences[key] = copy.deepcopy(
                stored_preferences[key]
            )

    return effective_profile