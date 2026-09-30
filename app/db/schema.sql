-- ============================================================
-- SafeSight AI — Production PostgreSQL Schema
-- Human-in-the-Loop (HITL) Moderation Pipeline
-- ============================================================
-- Compatible with:  PostgreSQL 14+  |  SQLite 3.35+ (with minor adjustments noted)
-- Encoding:         UTF-8
-- Timezone:         All timestamps stored in UTC
-- ============================================================

-- ── Extensions (PostgreSQL only — remove for SQLite) ────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- For gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- For fuzzy text search on content

-- ─────────────────────────────────────────────────────────────────────────────
-- ENUM TYPES
-- ─────────────────────────────────────────────────────────────────────────────

-- Core moderation decisions
CREATE TYPE moderation_decision AS ENUM (
    'APPROVED',
    'FLAGGED',
    'HUMAN_AUDIT_REQUIRED'
);

-- Violation categories (mirrors app/schemas.py ViolationCategory)
CREATE TYPE violation_category AS ENUM (
    'HATE_SPEECH_AND_HARASSMENT',
    'SELF_HARM',
    'EXPLICIT_CONTENT',
    'DEEPFAKE',
    'CYBERBULLYING',
    'VIOLENT_THREATS',
    'MISINFORMATION',
    'SPAM_AND_MANIPULATION'
);

-- Severity levels
CREATE TYPE severity_level AS ENUM ('HIGH', 'MEDIUM', 'LOW');

-- Pipeline tier used
CREATE TYPE pipeline_tier AS ENUM ('TIER1', 'TIER2');

-- Human moderator audit statuses
CREATE TYPE audit_status AS ENUM (
    'PENDING',             -- Awaiting assignment to a moderator
    'ASSIGNED',            -- Assigned to a specific moderator, not yet reviewed
    'APPROVED_BY_HUMAN',   -- Moderator agrees: content is safe
    'BLOCKED_BY_HUMAN',    -- Moderator agrees: content should be removed
    'ESCALATED',           -- Requires senior review / legal
    'TIMED_OUT'            -- No moderator action within SLA window
);

-- Media types processed
CREATE TYPE media_type AS ENUM ('text', 'image', 'audio', 'video');


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 1: moderation_logs
-- Immutable audit log of every moderation decision made by the AI pipeline.
-- Never updated — only INSERT. Deletions only via data-retention jobs.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS moderation_logs (
    -- Primary key
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ── Request context ───────────────────────────────────────────────────────
    request_id          VARCHAR(64),                    -- X-Request-ID from API middleware
    session_id          VARCHAR(128),                   -- Optional: upstream session/user token (hashed)
    platform_source     VARCHAR(64),                    -- e.g. 'web', 'mobile_ios', 'api_v1'

    -- ── Content ───────────────────────────────────────────────────────────────
    original_text       TEXT,                           -- Raw (pre-preprocessing) text submitted
    cleaned_text        TEXT,                           -- Text after adversarial preprocessing
    image_file_path     TEXT,                           -- Relative path or object store key (S3/GCS)
    image_dhash         VARCHAR(64),                    -- Perceptual dHash fingerprint (hex string)
    audio_file_path     TEXT,                           -- Audio file path if audio was submitted
    audio_transcript    TEXT,                           -- Whisper transcription of audio (if any)
    media_types         media_type[],                   -- Array of modalities processed

    -- ── AI Decision ───────────────────────────────────────────────────────────
    model_decision      moderation_decision NOT NULL,
    confidence_score    NUMERIC(5, 4) NOT NULL          -- e.g. 0.9412
                            CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    pipeline_tier       pipeline_tier NOT NULL,
    total_latency_ms    NUMERIC(10, 2),                 -- End-to-end pipeline latency

    -- ── Violation details (JSON array of DetectedViolation objects) ───────────
    detected_violations JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- ── Explainability ────────────────────────────────────────────────────────
    raw_model_json      JSONB NOT NULL,                 -- Full ModerationResult JSON payload
    text_explanation    TEXT,                           -- Extracted reasoning.text_explanation
    visual_explanation  TEXT,                           -- Extracted reasoning.visual_explanation
    bimodal_synergy     TEXT,                           -- Extracted reasoning.bimodal_synergy

    -- ── Preprocessing flags ───────────────────────────────────────────────────
    leet_detected       BOOLEAN NOT NULL DEFAULT FALSE, -- Leet-speak was found and normalized
    spaced_text_detected BOOLEAN NOT NULL DEFAULT FALSE,-- Artificially spaced text was found
    banned_hash_match   BOOLEAN NOT NULL DEFAULT FALSE, -- Image matched banned-hash database

    -- ── Timestamps ────────────────────────────────────────────────────────────
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_date        TIMESTAMPTZ,                    -- When content was originally posted (if known)

    -- ── Soft delete ───────────────────────────────────────────────────────────
    deleted_at          TIMESTAMPTZ                     -- NULL = active, set = soft-deleted
);

-- Indexes for moderation_logs
CREATE INDEX IF NOT EXISTS idx_modlogs_decision     ON moderation_logs (model_decision);
CREATE INDEX IF NOT EXISTS idx_modlogs_confidence   ON moderation_logs (confidence_score);
CREATE INDEX IF NOT EXISTS idx_modlogs_created_at   ON moderation_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_modlogs_dhash        ON moderation_logs (image_dhash)
    WHERE image_dhash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_modlogs_request_id   ON moderation_logs (request_id)
    WHERE request_id IS NOT NULL;
-- Full-text index on text content for moderator search
CREATE INDEX IF NOT EXISTS idx_modlogs_text_gin     ON moderation_logs
    USING gin (to_tsvector('english', COALESCE(original_text, '')));
-- JSONB index for violation queries
CREATE INDEX IF NOT EXISTS idx_modlogs_violations   ON moderation_logs
    USING gin (detected_violations);

COMMENT ON TABLE moderation_logs IS
    'Immutable append-only audit log of every AI moderation decision. '
    'Do not UPDATE rows — use override_logs for corrections.';


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 2: audit_queue
-- Human moderator work queue. Populated when:
--   (a) model_decision = 'HUMAN_AUDIT_REQUIRED', OR
--   (b) confidence_score falls in the ambiguous band [0.35, 0.75]
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_queue (
    -- Primary key
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ── Link to source moderation log ─────────────────────────────────────────
    moderation_log_id   UUID NOT NULL
                            REFERENCES moderation_logs(id) ON DELETE CASCADE,

    -- ── Routing metadata ──────────────────────────────────────────────────────
    queue_reason        VARCHAR(128) NOT NULL,   -- e.g. 'LOW_CONFIDENCE', 'AUDIT_REQUIRED', 'BANNED_HASH_NEAR_MATCH'
    priority            SMALLINT NOT NULL DEFAULT 5
                            CHECK (priority BETWEEN 1 AND 10),
                            -- 1 = urgent (CSAM/violence), 10 = routine
    sla_deadline        TIMESTAMPTZ,            -- Must be resolved by this time

    -- ── Assignment ────────────────────────────────────────────────────────────
    status              audit_status NOT NULL DEFAULT 'PENDING',
    moderator_id        VARCHAR(128),           -- Internal employee ID or hashed identifier
    assigned_at         TIMESTAMPTZ,            -- When this item was picked up by a moderator
    reviewed_at         TIMESTAMPTZ,            -- When the moderator submitted their decision

    -- ── Human decision ────────────────────────────────────────────────────────
    human_decision      moderation_decision,    -- What the human concluded
    moderator_notes     TEXT,                   -- Free-text explanation from the moderator
    resolution_tags     TEXT[],                 -- e.g. ARRAY['satire', 'false_positive', 'borderline']

    -- ── SLA tracking ──────────────────────────────────────────────────────────
    escalation_count    SMALLINT NOT NULL DEFAULT 0,  -- How many times escalated
    timed_out           BOOLEAN NOT NULL DEFAULT FALSE,

    -- ── Timestamps ────────────────────────────────────────────────────────────
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at on any row change
CREATE OR REPLACE FUNCTION update_audit_queue_timestamp()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_audit_queue_updated_at
    BEFORE UPDATE ON audit_queue
    FOR EACH ROW EXECUTE FUNCTION update_audit_queue_timestamp();

-- Indexes for audit_queue
CREATE INDEX IF NOT EXISTS idx_auditq_status        ON audit_queue (status)
    WHERE status = 'PENDING';                  -- Partial index — only pending items
CREATE INDEX IF NOT EXISTS idx_auditq_moderator     ON audit_queue (moderator_id)
    WHERE moderator_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_auditq_priority      ON audit_queue (priority, created_at)
    WHERE status IN ('PENDING', 'ASSIGNED');
CREATE INDEX IF NOT EXISTS idx_auditq_sla           ON audit_queue (sla_deadline)
    WHERE status NOT IN ('APPROVED_BY_HUMAN', 'BLOCKED_BY_HUMAN', 'TIMED_OUT');
CREATE INDEX IF NOT EXISTS idx_auditq_log_id        ON audit_queue (moderation_log_id);

COMMENT ON TABLE audit_queue IS
    'Human moderator work queue. Rows inserted automatically when AI confidence '
    'is in the ambiguous band [0.35, 0.75] or decision is HUMAN_AUDIT_REQUIRED. '
    'SLA deadline enforced by a background worker.';


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 3: override_logs
-- Captures every instance where a human moderator reversed an AI decision.
-- Critical dataset for generating fine-tuning examples and measuring AI accuracy.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS override_logs (
    -- Primary key
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ── Links ─────────────────────────────────────────────────────────────────
    moderation_log_id       UUID NOT NULL
                                REFERENCES moderation_logs(id) ON DELETE CASCADE,
    audit_queue_id          UUID
                                REFERENCES audit_queue(id) ON DELETE SET NULL,

    -- ── The override ──────────────────────────────────────────────────────────
    original_ai_decision    moderation_decision NOT NULL,   -- What the AI said
    original_confidence     NUMERIC(5, 4) NOT NULL,         -- AI's confidence
    human_override_decision moderation_decision NOT NULL,   -- What the human decided
    override_direction      VARCHAR(32) NOT NULL            -- 'FALSE_POSITIVE' | 'FALSE_NEGATIVE' | 'ESCALATION'
                                CHECK (override_direction IN (
                                    'FALSE_POSITIVE',   -- AI flagged; human approved
                                    'FALSE_NEGATIVE',   -- AI approved; human blocked
                                    'ESCALATION',       -- Human sent to senior/legal
                                    'SEVERITY_CHANGE'   -- Same decision but severity changed
                                )),

    -- ── Explanation ───────────────────────────────────────────────────────────
    moderator_id            VARCHAR(128) NOT NULL,
    override_reason         TEXT NOT NULL,              -- Required justification
    policy_reference        VARCHAR(64),                -- e.g. 'MLC_H1.1' — which rule applies
    contributing_factors    TEXT[],                     -- e.g. ARRAY['satire', 'cultural_context', 'missing_audio']

    -- ── Fine-tuning metadata ──────────────────────────────────────────────────
    added_to_finetune_set   BOOLEAN NOT NULL DEFAULT FALSE,
    finetune_label          moderation_decision,        -- Label to use for fine-tuning
    finetune_exported_at    TIMESTAMPTZ,                -- When this row was exported to training data

    -- ── Timestamps ────────────────────────────────────────────────────────────
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for override_logs
CREATE INDEX IF NOT EXISTS idx_overrides_direction  ON override_logs (override_direction);
CREATE INDEX IF NOT EXISTS idx_overrides_moderator  ON override_logs (moderator_id);
CREATE INDEX IF NOT EXISTS idx_overrides_finetune   ON override_logs (added_to_finetune_set)
    WHERE added_to_finetune_set = FALSE;       -- Easy query for un-exported rows
CREATE INDEX IF NOT EXISTS idx_overrides_log_id     ON override_logs (moderation_log_id);
CREATE INDEX IF NOT EXISTS idx_overrides_created    ON override_logs (created_at DESC);

COMMENT ON TABLE override_logs IS
    'Every human reversal of an AI decision. Used for: '
    '(1) accuracy metrics, (2) fine-tuning dataset generation, '
    '(3) policy calibration. Never delete rows from this table.';


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 4: banned_image_hashes (operational reference table)
-- Persistent store of known-bad perceptual hashes.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS banned_image_hashes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dhash           VARCHAR(64) NOT NULL UNIQUE,
    hash_type       VARCHAR(16) NOT NULL DEFAULT 'dhash8',  -- 'dhash8', 'phash', 'ahash'
    category        violation_category NOT NULL,
    severity        severity_level NOT NULL DEFAULT 'HIGH',
    source          VARCHAR(128),       -- e.g. 'NCMEC', 'internal_review', 'partner_feed'
    description     TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    added_by        VARCHAR(128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ         -- NULL = permanent ban
);

CREATE INDEX IF NOT EXISTS idx_banned_dhash    ON banned_image_hashes (dhash) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_banned_category ON banned_image_hashes (category) WHERE is_active = TRUE;

COMMENT ON TABLE banned_image_hashes IS
    'Persistent store of known-bad perceptual image hashes. '
    'The application preprocessor checks uploads against this table '
    'before invoking any ML model (instant bypass for known content).';


-- ─────────────────────────────────────────────────────────────────────────────
-- VIEWS
-- ─────────────────────────────────────────────────────────────────────────────

-- Moderator dashboard: pending items ordered by priority
CREATE OR REPLACE VIEW v_pending_audit_items AS
SELECT
    aq.id                   AS queue_id,
    aq.priority,
    aq.queue_reason,
    aq.sla_deadline,
    aq.created_at           AS queued_at,
    ml.model_decision,
    ml.confidence_score,
    ml.original_text,
    ml.image_file_path,
    ml.image_dhash,
    ml.detected_violations,
    ml.text_explanation,
    ml.visual_explanation,
    ml.bimodal_synergy,
    ml.platform_source,
    ml.created_at           AS content_submitted_at
FROM audit_queue aq
JOIN moderation_logs ml ON ml.id = aq.moderation_log_id
WHERE aq.status IN ('PENDING', 'ASSIGNED')
ORDER BY aq.priority ASC, aq.created_at ASC;

COMMENT ON VIEW v_pending_audit_items IS
    'Moderator dashboard view: all unresolved items ordered by priority then age.';


-- AI accuracy report: false positive/negative rates by day
CREATE OR REPLACE VIEW v_override_accuracy_daily AS
SELECT
    DATE_TRUNC('day', ol.created_at)    AS report_date,
    ol.override_direction,
    COUNT(*)                             AS override_count,
    AVG(ol.original_confidence)          AS avg_ai_confidence,
    COUNT(*) FILTER (WHERE ol.added_to_finetune_set) AS finetune_ready
FROM override_logs ol
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

COMMENT ON VIEW v_override_accuracy_daily IS
    'Daily breakdown of AI override rates for accuracy reporting and SLO tracking.';


-- ─────────────────────────────────────────────────────────────────────────────
-- FUNCTION: auto_route_to_audit_queue
-- Trigger function: automatically inserts into audit_queue when a moderation_log
-- row has decision = HUMAN_AUDIT_REQUIRED or confidence in [0.35, 0.75].
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION auto_route_to_audit_queue()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
    v_reason    VARCHAR(128);
    v_priority  SMALLINT;
    v_sla       TIMESTAMPTZ;
BEGIN
    -- Determine if this log entry should enter the audit queue
    IF NEW.model_decision = 'HUMAN_AUDIT_REQUIRED' THEN
        v_reason   := 'AUDIT_REQUIRED';
        v_priority := 5;
        v_sla      := NOW() + INTERVAL '8 hours';

    ELSIF NEW.confidence_score BETWEEN 0.35 AND 0.75 THEN
        v_reason   := 'LOW_CONFIDENCE';
        v_priority := 6;
        v_sla      := NOW() + INTERVAL '24 hours';

    ELSIF NEW.banned_hash_match = TRUE AND NEW.model_decision != 'FLAGGED' THEN
        -- Near-match to banned hash but AI didn't flag — needs human review
        v_reason   := 'BANNED_HASH_NEAR_MATCH';
        v_priority := 2;
        v_sla      := NOW() + INTERVAL '2 hours';

    ELSE
        -- No audit needed
        RETURN NEW;
    END IF;

    INSERT INTO audit_queue (
        moderation_log_id,
        queue_reason,
        priority,
        sla_deadline
    ) VALUES (
        NEW.id,
        v_reason,
        v_priority,
        v_sla
    );

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_auto_audit_route
    AFTER INSERT ON moderation_logs
    FOR EACH ROW EXECUTE FUNCTION auto_route_to_audit_queue();

COMMENT ON FUNCTION auto_route_to_audit_queue() IS
    'Automatically routes moderation log entries to the human audit queue '
    'based on AI decision and confidence thresholds.';


-- ─────────────────────────────────────────────────────────────────────────────
-- SQLITE COMPATIBILITY NOTES
-- ─────────────────────────────────────────────────────────────────────────────
-- If using SQLite instead of PostgreSQL:
-- 1. Remove CREATE EXTENSION statements
-- 2. Replace UUID PRIMARY KEY DEFAULT gen_random_uuid() with:
--    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16))))
-- 3. Replace ENUM types with TEXT + CHECK constraints
-- 4. Replace TIMESTAMPTZ with TEXT (store ISO 8601 strings)
-- 5. Replace JSONB with TEXT (store JSON strings)
-- 6. Remove triggers and views (implement in application layer)
-- 7. Replace NUMERIC(5,4) with REAL
-- ─────────────────────────────────────────────────────────────────────────────
