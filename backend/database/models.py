from __future__ import annotations

import psycopg2.extensions

_DDL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS datasets (
    id            UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT      NOT NULL,
    rows          INTEGER   NOT NULL,
    columns       INTEGER   NOT NULL,
    target_column TEXT,
    excluded_columns JSONB  NOT NULL DEFAULT '[]'::jsonb,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

ALTER TABLE datasets
ADD COLUMN IF NOT EXISTS excluded_columns JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS dataset_features (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id      UUID    NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    num_samples     INTEGER NOT NULL,
    num_features    INTEGER NOT NULL,
    num_numeric     INTEGER NOT NULL,
    num_categorical INTEGER NOT NULL,
    missing_ratio   FLOAT   NOT NULL,
    imbalance_ratio FLOAT   NOT NULL,
    UNIQUE (dataset_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id                 UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id         UUID  NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    recommended_models JSONB NOT NULL,
    reasoning          TEXT  NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id            UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id    UUID      NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    model_name    TEXT      NOT NULL,
    accuracy      FLOAT     NOT NULL,
    precision     FLOAT     NOT NULL,
    recall        FLOAT     NOT NULL,
    f1_score      FLOAT     NOT NULL,
    training_time FLOAT     NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id            UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id    UUID      NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    model_name    TEXT      NOT NULL,
    artifact_path TEXT      NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_base_entries (
    id                          UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id                  UUID      NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    dataset_name                TEXT      NOT NULL,
    target_column               TEXT      NOT NULL,
    recommended_models          JSONB     NOT NULL,
    reasoning                   TEXT      NOT NULL,
    system_guidance             JSONB     NOT NULL,
    best_model                  TEXT      NOT NULL,
    best_accuracy               FLOAT     NOT NULL,
    top_recommendation_model    TEXT,
    top_recommendation_worked   BOOLEAN   NOT NULL DEFAULT FALSE,
    experiment_count            INTEGER   NOT NULL,
    created_at                  TIMESTAMP NOT NULL DEFAULT NOW()
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()

# ---- legacy ORM classes removed; raw SQL is used throughout ----
