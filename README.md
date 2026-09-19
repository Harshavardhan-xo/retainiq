# RetainIQ
### SaaS Customer Churn & Revenue-at-Risk Dashboard

> **How to use this document:** Paste this whole file into ChatGPT as your
> first message and ask it to build the project exactly as specified below,
> file by file, in the order given in Section 11.

## 1. Business Problem
Subscription businesses lose recurring revenue to churn that's often visible
in the data weeks before a customer cancels — falling logins, rising support
tickets, no feature adoption. Customer Success teams need a ranked,
$-quantified list of *who* is at risk and *why*, not just a lagging
churn-rate metric.

**Business questions this project answers:**
- Which active accounts are most likely to churn in the next cycle?
- How much MRR is at risk right now?
- What's the single biggest risk driver per account, so CS knows what to fix
  first?

## 2. Business Impact
Turns a reactive "why did they leave" postmortem into a proactive "who do we
call this week" action list — the core value a BA/analyst adds in a Customer
Success or RevOps function.

## 3. Solution Overview
A local SQLite database of synthetic SaaS customers, usage, tickets, and
subscription status; SQL views for cohort retention; a Python-trained
logistic regression churn-risk score; a Streamlit dashboard ranking at-risk
MRR.

## 4. Tech Stack
| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core logic |
| Database | SQLite (`sqlite3` / SQLAlchemy) | Local relational store |
| Query layer | SQL | Cohort & RFM-style views |
| Modeling | scikit-learn (LogisticRegression) | Churn-risk scoring |
| Data handling | pandas, numpy | Transformation |
| App/dashboard | Streamlit | Interactive UI |
| Charts | Plotly | Retention curve, risk distribution |
| Testing | pytest | Unit tests |
| Version control | Git + GitHub | Source control |

## 5. Architecture / Pipeline
```
[generate_synthetic_data.py] --> [SQLite: retainiq.db]
        v
[queries/*.sql]  --(pandas.read_sql)-->  cohort & usage views
        v
[churn_model.py] -- feature engineering + LogisticRegression -->
        risk_score (0-1) per active customer
        v
[metrics.py] --> KPI summary (MRR at risk, churn rate, avg tenure)
        v
[app.py: Streamlit] --> KPI cards, retention curve, risk table, save-list
```
**Ingest (synthetic generator) → Store (SQLite) → Transform (SQL views) →
Model (scikit-learn) → Visualize (Streamlit) → Recommend (save-list)**

## 6. Data Model
**`customers`**
| Column | Type |
|---|---|
| customer_id | INTEGER PK |
| signup_date | DATE |
| plan_tier | TEXT (starter/pro/enterprise) |
| mrr | REAL |
| industry | TEXT |
| region | TEXT |

**`usage_events`**
| Column | Type |
|---|---|
| customer_id | INTEGER FK |
| event_date | DATE |
| feature_used | TEXT |
| login_count | INTEGER |

**`support_tickets`**
| Column | Type |
|---|---|
| customer_id | INTEGER FK |
| ticket_date | DATE |
| category | TEXT (billing/bug/how_to/other) |
| resolved_flag | BOOLEAN |

**`subscriptions`**
| Column | Type |
|---|---|
| customer_id | INTEGER FK |
| status | TEXT (active/churned) |
| churn_date | DATE (nullable) |

## 7. Synthetic Data Generation Rules
- ~2,000 customers, signup dates spread over the last 24 months
- Build churn probability as a weighted logistic function of short tenure,
  low login_count, high ticket count, and `starter` tier, then sample
  churned/active from it, so the later model has real signal to find
- Generate 1–60 usage events and 0–8 tickets per customer, volume scaled
  inversely with churn risk (churners use the product less)
- Seed the random generator for reproducibility

## 8. Modeling Details
- Features: tenure_months, avg_monthly_logins, total_tickets,
  pct_tickets_unresolved, plan_tier (one-hot)
- Model: `sklearn.linear_model.LogisticRegression`, train/test split, log the
  AUC at training time
- Output: `risk_score` (0–1) per **active** customer only
- Store the feature with the largest positive coefficient contribution per
  row as `top_risk_driver` (for the save-list table)

## 9. Dashboard Specification
- **Sidebar:** risk-score threshold slider (default 0.5), plan-tier filter
- **KPI row:** total MRR, MRR at risk (sum of mrr where risk_score ≥
  threshold), churn rate (last 12 months), avg tenure
- **Chart 1:** cohort retention curve (% of each signup-month cohort still
  active, by months since signup)
- **Chart 2:** histogram of risk_score distribution across active customers
- **Table:** sortable "save list" — customer_id, plan_tier, mrr, risk_score,
  top_risk_driver, filtered by the sidebar threshold

## 10. File & Folder Structure
```
retainiq/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── generate_synthetic_data.py
├── tests/
│   ├── test_churn_model.py
│   └── test_metrics.py
├── queries/
│   ├── cohort_retention.sql
│   └── rfm_scoring.sql
└── src/
    ├── __init__.py
    ├── db.py
    ├── churn_model.py
    └── metrics.py
```

## 11. Step-by-Step Build Order
1. Scaffold folders; `.gitignore`, `LICENSE`.
2. Write `data/generate_synthetic_data.py` implementing Section 7; running it
   creates `retainiq.db` with all 4 tables populated.
3. Write `src/db.py`: connection helper + typed query functions wrapping the
   `.sql` files in `queries/`.
4. Write the two `.sql` files (cohort retention by signup month; an
   RFM-style recency/frequency/monetary score per customer).
5. Write `src/churn_model.py`: feature engineering + `train_and_score() ->
   DataFrame` per Section 8.
6. Write `src/metrics.py`: `summarize(customers_df, scored_df) -> dict`.
7. Write `tests/`: assert risk_score is in [0,1] for every active customer,
   and that a synthetic "always logs in, never tickets" customer scores
   lower than a "no logins, many tickets" one.
8. Write `app.py` wiring db → churn_model → metrics → dashboard (Section 9).
9. Run locally, fix all exceptions.
10. Write final `README.md` per Section 14.
11. `git init`, commit, push.

## 12. Production-Quality Bar
- [ ] Type hints + docstrings on every function
- [ ] No bare `except:`
- [ ] DB schema created via a single idempotent `init_db()` (safe to re-run)
- [ ] `logging` for the model training step (log AUC, row counts)
- [ ] `pytest` passes
- [ ] `requirements.txt` pinned
- [ ] Zero unhandled exceptions in the app

## 13. Roadmap / Future Enhancements
- Swap SQLite for a Postgres connection string (same SQL, one config change)
  to show data-warehouse readiness
- Add a second model (gradient boosting) and compare AUC
- Recreate the save-list view in Power BI/Tableau connected to the same
  SQLite file
- Add a "what-if" slider: simulate a login increase and show projected
  risk-score change

## 14. Required Contents of Final `README.md`
Business problem → schema summary → tech stack → setup steps → how the risk
score is computed (plain language) → limitations (synthetic-data disclosure,
not a production credit-risk-grade model) → screenshots placeholder.

## 15. Definition of Done
- [ ] `generate_synthetic_data.py` produces a working `retainiq.db` in one run
- [ ] Dashboard runs with zero exceptions
- [ ] Save-list correctly re-sorts and re-filters on threshold change
- [ ] Tests pass
- [ ] README complete

## 16. Resume Bullet Template
"Built an end-to-end SaaS churn-prediction pipeline (SQL, Python/scikit-learn,
Streamlit) on a synthetic 2,000-customer dataset, surfacing $[X] in at-risk
MRR and each account's top churn driver."