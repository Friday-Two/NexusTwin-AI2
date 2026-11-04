# Phase 10 — Persistent Digital Twin Memory, Preferences and Consent

## 1. Overview

Phase 10 introduces a persistent, user-specific memory and preference management system for the Personal AI Digital Twin.

The objective is to enable the Digital Twin to retain user-approved preferences across sessions while respecting explicit consent, user isolation, privacy restrictions and existing hard constraints.

The memory subsystem is implemented separately from the Phase 7 booking verification database.

## 2. Architecture

The implementation contains three primary modules:

- `store.py`: persistent memory storage, retrieval, deletion, consent enforcement and audit logging.
- `preferences.py`: structured preference storage, retrieval, updating and deletion.
- `profile_integration.py`: merges consented soft preferences into an effective application profile.

Application integration is performed in `src/main.py`.

The application obtains the user identity through the `DIGITAL_TWIN_USER_ID` environment variable.

If the user ID is missing, unregistered or lacks preference retrieval consent, the original profile is retained.

## 3. Database Design

Database location:

`digital_twin/memory/digital_twin_memory.db`

Schema location:

`digital_twin/memory/schema.sql`

The database contains five tables:

| Table | Responsibility |
|---|---|
| `memory_users` | Registered user identities |
| `memories` | Persistent user memory records |
| `preferences` | Structured user preferences |
| `consents` | User permissions by consent scope |
| `memory_audit` | Records memory and preference operations |

Four indexes support user-specific lookup and audit queries.

### Database constraints

- Foreign keys associate records with registered users.
- Preference keys are unique per user.
- Consent scopes are unique per user.
- Consent values are restricted to 0 or 1.
- Memory sensitivity is restricted to normal or sensitive.
- Memory status is restricted to active, deleted or expired.

SQLite foreign-key enforcement is enabled for each connection by the memory store.

## 4. Consent Management

Consent is maintained independently for each user and operation scope.

Implemented consent scopes:

- `memory_storage`
- `memory_retrieval`
- `preference_storage`
- `preference_retrieval`

Consent is not automatically granted.

Storage and retrieval operations require their corresponding permissions.

Revoking retrieval consent prevents stored information from being retrieved through the relevant API.

Deletion operations remain available without retrieval consent.

## 5. Persistent Memory Management

The memory store supports:

- User registration.
- Memory creation.
- Consent-checked retrieval.
- Category-based filtering.
- Memory deletion.
- Expiration metadata.
- Audit logging.

Sensitive memory storage is rejected by the current implementation.

Audit records identify operations without storing the deleted memory content.

## 6. Preference Management

The preference manager supports structured JSON-compatible values.

Operations include:

- Save a preference.
- Retrieve preferences.
- Update an existing preference.
- Delete a preference.
- Enforce storage and retrieval consent.
- Audit preference operations.

A unique database constraint prevents duplicate preference keys for the same user.

## 7. Profile Integration

The `build_effective_profile()` function combines the original profile with consented persistent preferences.

The currently permitted soft-preference keys are:

- `prefer_lower_price`
- `preferred_amenities`
- `preferred_hotel_type`

The original profile is copied before modification.

Persistent preferences cannot override hard constraints or privacy settings.

Protected profile fields include:

- Maximum total price.
- Hotel verification requirement.
- Phone-sharing privacy policy.

If consent is unavailable, the original profile is returned.

## 8. Application Integration

The application reads the identity from:

`DIGITAL_TWIN_USER_ID`

The environment variable is optional.

The original profile is loaded first, then passed through `build_effective_profile()` before request processing.

The existing Phase 7 execution and verification logic was not redesigned as part of Phase 10.

The complete `main.py` execution was intentionally not used for this validation because its fixed run identifier, `phase7-demo-001`, could overwrite existing Phase 7 evidence.

Profile integration was instead validated independently using the real profile and consented demonstration user.

## 9. Security and Privacy Safeguards

The implementation includes:

1. User-specific record ownership.
2. Explicit, scope-specific consent.
3. Consent checks before storage and retrieval.
4. User isolation.
5. Sensitive-memory rejection.
6. Audit logging.
7. Preference allowlisting during profile integration.
8. Protection of hard constraints.
9. Protection of existing privacy settings.
10. Copy-based profile integration to avoid modifying the original profile.

## 10. Validation Results

### Automated tests

| Test module | Tests | Result |
|---|---:|---|
| Phase 8 Digital Twin | 20 | Passed |
| Phase 8 Execution Verifier | 8 | Passed |
| Phase 9 Allocation Validator | 9 | Passed |
| Phase 9 AI Inference | 9 | Passed |
| Phase 10 Memory | 12 | Passed |
| Phase 10 Preferences | 10 | Passed |
| Phase 10 Profile Integration | 8 | Passed |
| **Total** | **76** | **Passed** |

Full regression result:

`Ran 76 tests in 3.725s — OK`

### Manual integration verification

The following scenarios were demonstrated:

- Registration of a separate demonstration user.
- Explicit granting of preference storage and retrieval consent.
- Saving and retrieving `preferred_hotel_type = budget`.
- Merging the preference into the effective profile.
- Preserving the existing `prefer_lower_price` preference.
- Preserving the maximum budget of INR 6,000.
- Confirming that the original profile remains unchanged.
- Revoking retrieval consent.
- Confirming fallback to the original profile after consent revocation.

### Database verification

SQLite integrity check: `ok`

Foreign-key violations: `0`

Expected tables present: `5`

## 11. Known Limitations

- The complete application startup and booking execution flow has not been rerun with Phase 10 enabled, to protect Phase 7 evidence.
- The current integration supports an allowlist of three soft-preference keys.
- The user identity is supplied through an environment variable; a production authentication-to-memory identity integration remains a separate engineering task.
- Sensitive memory storage is rejected rather than supported through an encrypted sensitive-memory workflow.
- Expiration metadata is supported, but a background expiration and cleanup scheduler has not been implemented.
- The Phase 9 FLAN-T5 model's allocation reliability remains a separate concern and is not improved by Phase 10.

## 12. Conclusion

Phase 10 implements a persistent, user-specific memory and preference subsystem with explicit consent, user isolation, audit logging and profile integration.

The implementation passed all 76 tests in the combined regression suite.

Manual verification demonstrated that consented preferences influence soft profile preferences without changing hard constraints or privacy settings.

The subsystem establishes the foundation for a persistent, preference-aware Personal AI Digital Twin.
