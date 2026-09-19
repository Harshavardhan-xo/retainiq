# RetainIQ

### Customer Retention Command Center

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://harsha-retainiq.streamlit.app)

**[Open Live Dashboard ↗](https://harsha-retainiq.streamlit.app)**

## What this demonstrates

A production-style Customer Success / RevOps analytics workflow built around a **15,000-customer synthetic SaaS portfolio** and **180,000 monthly activity observations**.

- Account-level churn-risk scoring with Logistic Regression
- Model validation with held-out AUC
- MRR-at-risk and prioritized save-list
- Cohort retention and plan-level benchmarking
- Adoption, NPS, ticket and expansion signals
- Portfolio filters, downloadable CSV output and executive KPIs

## Technology

Python • Streamlit • pandas • NumPy • scikit-learn • Plotly

## Dashboard workflow

**Portfolio → Segment → Risk → Revenue exposure → Save list**

## Data disclosure

The data is synthetic and designed for portfolio demonstration. The model is illustrative and is not a production credit-risk or customer-decisioning system.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
