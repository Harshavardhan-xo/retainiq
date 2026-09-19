# RetainIQ
### SaaS Customer Churn & Revenue-at-Risk Dashboard

End-to-end churn analytics using synthetic SaaS customer, usage, ticket, and subscription data. The pipeline loads SQLite data, builds features with SQL, trains a logistic regression model, and exposes a Streamlit save-list ranked by risk and MRR.

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
python data/generate_synthetic_data.py
streamlit run app.py
pytest -q
```

## Business questions
- Which active accounts are most likely to churn?
- How much MRR is at risk?
- What is the top risk driver for each account?

## Limitations
Synthetic data only. The risk score is illustrative, not production-grade; real deployment requires leakage controls, calibration, monitoring, and validation against true outcomes.