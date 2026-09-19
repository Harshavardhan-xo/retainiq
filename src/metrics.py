def summarize(customers, scored, threshold=.5):
    risk = scored[scored.risk_score >= threshold]
    return {
        "total_mrr": float(customers.loc[customers.churned==0, "mrr"].sum()),
        "mrr_at_risk": float(risk.mrr.sum()),
        "churn_rate": float(customers.churned.mean()),
        "avg_tenure_months": float(customers.tenure_months.mean()),
    }