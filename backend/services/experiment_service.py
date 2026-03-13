import psycopg2.extensions
import psycopg2.extras


def replace_experiments(
    conn: psycopg2.extensions.connection,
    dataset_id_str: str,
    results: list[dict],
) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("DELETE FROM experiments WHERE dataset_id = %s", (dataset_id_str,))
        inserted: list[dict] = []
        for row in results:
            cur.execute(
                """
                INSERT INTO experiments
                    (dataset_id, model_name, accuracy, precision, recall, f1_score, training_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    dataset_id_str,
                    str(row["model_name"]),
                    float(row["accuracy"]),
                    float(row["precision"]),
                    float(row["recall"]),
                    float(row["f1_score"]),
                    float(row["training_time"]),
                ),
            )
            inserted.append(dict(cur.fetchone()))
    conn.commit()
    return inserted


def get_experiments(
    conn: psycopg2.extensions.connection,
    dataset_id_str: str,
) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM experiments WHERE dataset_id = %s ORDER BY created_at ASC",
            (dataset_id_str,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_dashboard_summary(conn: psycopg2.extensions.connection) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*)::int AS count FROM datasets")
        datasets_processed = int(cur.fetchone()["count"])

        cur.execute("SELECT COUNT(*)::int AS count FROM experiments")
        models_tested = int(cur.fetchone()["count"])

        # Approximate experiment runs as one run per dataset that has experiment rows.
        cur.execute("SELECT COUNT(DISTINCT dataset_id)::int AS count FROM experiments")
        experiments_run = int(cur.fetchone()["count"])

        cur.execute("SELECT COALESCE(MAX(accuracy), 0)::float AS best_accuracy FROM experiments")
        best_accuracy = float(cur.fetchone()["best_accuracy"])

        cur.execute(
            """
            SELECT model_name, AVG(accuracy)::float AS average_accuracy
            FROM experiments
            GROUP BY model_name
            ORDER BY average_accuracy DESC
            LIMIT 8
            """
        )
        model_performance = [dict(row) for row in cur.fetchall()]

    return {
        "datasets_processed": datasets_processed,
        "experiments_run": experiments_run,
        "models_tested": models_tested,
        "best_accuracy": best_accuracy,
        "model_performance": model_performance,
    }


def replace_model_artifacts(
    conn: psycopg2.extensions.connection,
    dataset_id_str: str,
    artifacts: list[dict[str, str]],
) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("DELETE FROM model_artifacts WHERE dataset_id = %s", (dataset_id_str,))
        saved: list[dict] = []
        for item in artifacts:
            cur.execute(
                """
                INSERT INTO model_artifacts (dataset_id, model_name, artifact_path)
                VALUES (%s, %s, %s)
                RETURNING *
                """,
                (dataset_id_str, item["model_name"], item["artifact_path"]),
            )
            saved.append(dict(cur.fetchone()))
    conn.commit()
    return saved


def get_model_artifact(conn: psycopg2.extensions.connection, artifact_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM model_artifacts WHERE id = %s", (artifact_id,))
        row = cur.fetchone()
    return dict(row) if row else None
