from __future__ import annotations

import json

import psycopg2.extensions
import psycopg2.extras

RULEBOOK: list[str] = [
    "Use tree ensembles like Random Forest and XGBoost as strong defaults for mixed tabular data.",
    "Use Logistic Regression when interpretability and fast iteration matter.",
    "Use SVM when feature space is high-dimensional and dataset size is moderate.",
    "High class imbalance should bias recommendations toward models robust to skew and weighted metrics.",
    "Numeric targets with very high cardinality should be treated carefully; if using classification, bin them into stable groups.",
    "Past training outcomes should influence future confidence in recommendations for similar data profiles.",
]


def get_rulebook() -> list[str]:
    return RULEBOOK


def create_knowledge_base_entry(
    conn: psycopg2.extensions.connection,
    *,
    dataset_id: str,
    dataset_name: str,
    target_column: str,
    recommended_models: list[str],
    reasoning: str,
    best_model: str,
    best_accuracy: float,
    experiment_count: int,
) -> dict:
    top_recommendation_model = recommended_models[0] if recommended_models else None
    top_recommendation_worked = best_model == top_recommendation_model if top_recommendation_model else False

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO knowledge_base_entries (
                dataset_id, dataset_name, target_column, recommended_models,
                reasoning, system_guidance, best_model, best_accuracy,
                top_recommendation_model, top_recommendation_worked, experiment_count
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                dataset_id,
                dataset_name,
                target_column,
                json.dumps(recommended_models),
                reasoning,
                json.dumps(RULEBOOK),
                best_model,
                best_accuracy,
                top_recommendation_model,
                top_recommendation_worked,
                experiment_count,
            ),
        )
        row = dict(cur.fetchone())
    conn.commit()
    return row


def list_knowledge_base_entries(conn: psycopg2.extensions.connection) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM knowledge_base_entries ORDER BY created_at DESC"
        )
        rows = [dict(row) for row in cur.fetchall()]

    for row in rows:
        row["system_guidance"] = list(row.get("system_guidance") or [])
        row["recommended_models"] = list(row.get("recommended_models") or [])
    return rows


def get_latest_recommendation(conn: psycopg2.extensions.connection, dataset_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT recommended_models, reasoning
            FROM recommendations
            WHERE dataset_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (dataset_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None