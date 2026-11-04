PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memory_users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT,
    FOREIGN KEY (user_id) REFERENCES memory_users(user_id),
    CHECK (sensitivity IN ('normal', 'sensitive')),
    CHECK (status IN ('active', 'deleted', 'expired'))
);

CREATE TABLE IF NOT EXISTS preferences (
    preference_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES memory_users(user_id),
    UNIQUE (user_id, preference_key)
);

CREATE TABLE IF NOT EXISTS consents (
    consent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    consent_scope TEXT NOT NULL,
    granted INTEGER NOT NULL DEFAULT 0,
    policy_version TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES memory_users(user_id),
    UNIQUE (user_id, consent_scope),
    CHECK (granted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS memory_audit (
    audit_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES memory_users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_user_status
ON memories(user_id, status);

CREATE INDEX IF NOT EXISTS idx_preferences_user
ON preferences(user_id);

CREATE INDEX IF NOT EXISTS idx_consents_user_scope
ON consents(user_id, consent_scope);

CREATE INDEX IF NOT EXISTS idx_memory_audit_user_time
ON memory_audit(user_id, created_at);
