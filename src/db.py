from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "retainiq.db"

def con(path=DEFAULT_DB):
    if path is None:
        path = DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError("Run data/generate_synthetic_data.py first.")
    return sqlite3.connect(path)

def load_features(path=DEFAULT_DB):
    sql = """WITH a AS (
      SELECT customer_id,
             SUM(CASE WHEN event_date>=date('now','-90 day') THEN login_count ELSE 0 END) last90
      FROM usage_events GROUP BY customer_id
    ),
    t AS (
      SELECT customer_id, COUNT(*) total_tickets,
             AVG(CASE WHEN resolved_flag=0 THEN 1.0 ELSE 0 END) unresolved
      FROM support_tickets GROUP BY customer_id
    )
    SELECT c.*,
           CAST((julianday('now')-julianday(c.signup_date))/30.44 AS REAL) tenure_months,
           COALESCE(a.last90,0)/3.0 avg_monthly_logins,
           COALESCE(t.total_tickets,0) total_tickets,
           COALESCE(t.unresolved,0) pct_tickets_unresolved,
           CASE WHEN s.status='churned' THEN 1 ELSE 0 END churned
    FROM customers c
    JOIN subscriptions s USING(customer_id)
    LEFT JOIN a USING(customer_id)
    LEFT JOIN t USING(customer_id);"""
    with con(path) as c:
        return pd.read_sql_query(sql, c)

def load_cohort(path=DEFAULT_DB):
    with con(path) as c:
        return pd.read_sql_query(
            (ROOT/"queries/cohort_retention.sql").read_text(), c
        )