from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "retainiq.db"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=24)
    ids = np.arange(1, n + 1)
    signup = pd.to_datetime(rng.integers(start.value, end.value, n))
    plan = rng.choice(["starter","pro","enterprise"], n, p=[.45,.4,.15])
    mrr = np.select(
        [plan=="starter", plan=="pro", plan=="enterprise"],
        [rng.uniform(49,149,n), rng.uniform(199,599,n), rng.uniform(800,2500,n)]
    )
    tenure = (end - signup).days / 30.44
    prob = sigmoid(-2.8 + .06*(12-np.minimum(tenure,12)) + .7*(plan=="starter") + rng.normal(0,.35,n))
    churn = rng.random(n) < prob

    churn_date = []
    for s, is_churn in zip(signup, churn):
        if is_churn:
            span = max((end-s).days, 45)
            churn_date.append((s + pd.Timedelta(days=int(rng.integers(30, span+1)))).date())
        else:
            churn_date.append(None)

    customers = pd.DataFrame({
        "customer_id":ids, "signup_date":signup.date, "plan_tier":plan,
        "mrr":mrr.round(2),
        "industry":rng.choice(["SaaS","Retail","Fintech","Healthcare"],n),
        "region":rng.choice(["India","US","UK","EU","APAC"],n),
    })
    subscriptions = pd.DataFrame({
        "customer_id":ids,
        "status":np.where(churn,"churned","active"),
        "churn_date":churn_date,
    })

    usage, tickets = [], []
    for cid, s, is_churn in zip(ids, signup, churn):
        days = max((end-s).days, 14)
        factor = .35 if is_churn else 1.0
        for d in rng.integers(1, days+1, int(rng.integers(1,61))):
            dt = s + pd.Timedelta(days=int(d))
            if dt <= end:
                usage.append([cid, dt.date(),
                               rng.choice(["core","reports","automation","integrations"]),
                               max(1,int(rng.poisson(5*factor+1)))])
        for _ in range(int(rng.integers(0,9))):
            dt = min(s + pd.Timedelta(days=int(rng.integers(1,days+1))), end)
            tickets.append([cid, dt.date(),
                            rng.choice(["billing","bug","how_to","other"], p=[.15,.35,.4,.1]),
                            bool(rng.random() > (.35 if is_churn else .15))])

    return (
        customers,
        pd.DataFrame(usage, columns=["customer_id","event_date","feature_used","login_count"]),
        pd.DataFrame(tickets, columns=["customer_id","ticket_date","category","resolved_flag"]),
        subscriptions,
    )

def init_db(path=DB_PATH):
    customers, usage, tickets, subscriptions = build()
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, signup_date DATE, plan_tier TEXT, mrr REAL, industry TEXT, region TEXT);
        CREATE TABLE usage_events(customer_id INTEGER, event_date DATE, feature_used TEXT, login_count INTEGER);
        CREATE TABLE support_tickets(customer_id INTEGER, ticket_date DATE, category TEXT, resolved_flag BOOLEAN);
        CREATE TABLE subscriptions(customer_id INTEGER PRIMARY KEY, status TEXT, churn_date DATE);
        """)
        customers.to_sql("customers", con, index=False, if_exists="append")
        usage.to_sql("usage_events", con, index=False, if_exists="append")
        tickets.to_sql("support_tickets", con, index=False, if_exists="append")
        subscriptions.to_sql("subscriptions", con, index=False, if_exists="append")

if __name__ == "__main__":
    init_db()
    print(DB_PATH)