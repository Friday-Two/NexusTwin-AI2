# Phase 10 — Faculty Demonstration Guide

## Project Title

Personal AI Digital Twin: Persistent Memory, Preference Management and Consent-Aware Personalization

## 1. Demonstration Objective

Demonstrate how the Digital Twin retains user-approved preferences across sessions and uses them to personalize an effective profile without violating privacy settings or hard constraints.

## 2. Demonstration Flow

### Step 1 — Explain the architecture

Show these files:

- digital_twin/memory/schema.sql
- digital_twin/src/memory/store.py
- digital_twin/src/memory/preferences.py
- digital_twin/src/memory/profile_integration.py
- digital_twin/src/main.py

Explain that the memory subsystem uses a separate database from the booking verification database.

### Step 2 — Explain the database

The five tables are:

1. memory_users
2. memories
3. preferences
4. consents
5. memory_audit

Explain user isolation and scope-specific consent.

### Step 3 — Demonstrate user registration

The demonstration identity is:

phase10-demo-user

It is separate from the earlier phase10-test-user.

### Step 4 — Demonstrate explicit consent

Explain the four consent scopes:

- memory_storage
- memory_retrieval
- preference_storage
- preference_retrieval

Preference storage and retrieval require explicit consent.

### Step 5 — Demonstrate preference persistence

The saved preference is:

    {
      "preferred_hotel_type": "budget"
    }

Explain that the preference is stored in SQLite and retrieved in a subsequent operation.

### Step 6 — Demonstrate profile integration

Original profile:

    {
      "preferences": {
        "prefer_lower_price": true
      },
      "max_total_price": 6000
    }

Effective profile:

    {
      "preferences": {
        "prefer_lower_price": true,
        "preferred_hotel_type": "budget"
      },
      "max_total_price": 6000
    }

Explain that the preference is merged while the original budget remains unchanged.

### Step 7 — Demonstrate consent revocation

Show that preference retrieval consent is revoked.

The effective profile then falls back to the original preferences.

Explain that revoking retrieval consent prevents stored preferences from influencing personalization. It does not automatically delete the stored preference.

### Step 8 — Demonstrate automated testing

Run:

    python -m unittest discover -s .\digital_twin\tests -p "test_*.py" -v

Expected final result:

    Ran 76 tests

    OK

The suite includes Phase 8, Phase 9 and Phase 10 tests.

### Step 9 — Demonstrate database integrity

Verified results:

- SQLite integrity: ok
- Foreign-key violations: 0
- Required tables: 5

## 3. Important Technical Concepts

### Persistent memory

Information is stored in SQLite and remains available across application sessions.

### Consent-based access

The system checks permission before storing or retrieving memories and preferences.

### Multi-user isolation

Records are associated with user IDs. One user's preferences must not be returned to another user.

### Soft preferences versus hard constraints

Soft preferences guide personalization. Hard constraints remain authoritative.

For example, a preference for budget hotels cannot increase the maximum permitted budget.

### Audit logging

Memory and preference operations are recorded so system activity can be inspected.

### Privacy protection

The original profile and privacy restrictions are preserved during profile integration.

## 4. Expected Faculty Questions

**Q1. Why do you need persistent memory?**

To retain user-approved preferences across sessions instead of asking the user to provide the same information repeatedly.

**Q2. How do you protect user privacy?**

The system uses user-specific records and explicit, scope-specific consent for storage and retrieval.

**Q3. What happens when consent is revoked?**

The corresponding operation is blocked. When preference retrieval consent is revoked, the original profile is used instead of stored preferences.

**Q4. Can memory override the user's budget?**

No. Only allowlisted soft preferences are merged. The original budget and other hard constraints remain authoritative.

**Q5. How do you support multiple users?**

Every memory, preference and consent record is associated with a user ID through database foreign keys.

**Q6. Why use SQLite?**

SQLite provides a lightweight persistent relational database suitable for this research prototype. A larger production deployment could use a server database.

**Q7. Has the complete booking application been tested with this integration?**

The profile integration has been independently validated, and the combined 76-test regression suite passes. The complete main.py booking flow has not been rerun because its fixed Phase 7 run ID could overwrite previously preserved evidence.

**Q8. What are the current limitations?**

The application obtains its user identity from an environment variable. Sensitive-memory storage, background expiration cleanup and full production authentication integration remain future work.

## 5. Faculty Demonstration Closing Statement

Phase 10 adds persistent, user-specific memory and preference management to the Personal AI Digital Twin. It uses explicit consent, user isolation and audit logging. The integration personalizes soft preferences without overriding hard constraints or privacy settings. The implementation passed all 76 tests in the combined regression suite.

## 6. Evidence Locations

- Engineering report: digital_twin/reports/phase10/PHASE10_ENGINEERING_REPORT.md
- Memory schema: digital_twin/memory/schema.sql
- Memory database: digital_twin/memory/digital_twin_memory.db
- Memory tests: digital_twin/tests/test_phase10_memory.py
- Preference tests: digital_twin/tests/test_phase10_preferences.py
- Profile integration tests: digital_twin/tests/test_phase10_profile_integration.py
