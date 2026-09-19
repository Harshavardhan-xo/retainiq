import streamlit as st
import plotly.express as px
from src.churn_model import train_and_score
from src.db import load_features, load_cohort

st.set_page_config(page_title="RetainIQ", layout="wide")
st.title("RetainIQ — Churn & Revenue at Risk")
try:
    scored, auc = train_and_score()
    customers = load_features()
except FileNotFoundError:
    st.error("Run python data/generate_synthetic_data.py first.")
    st.stop()

threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
plans = st.sidebar.multiselect("Plan tier", sorted(scored.plan_tier.unique()),
                               default=sorted(scored.plan_tier.unique()))
filtered = scored[scored.plan_tier.isin(plans)]
risk = filtered[filtered.risk_score >= threshold]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Active MRR", f"${customers.loc[customers.churned==0,'mrr'].sum():,.0f}")
c2.metric("MRR at Risk", f"${risk.mrr.sum():,.0f}")
c3.metric("Churn Rate", f"{customers.churned.mean():.1%}")
c4.metric("Avg Tenure", f"{customers.tenure_months.mean():.1f} mo")

left,right = st.columns(2)
with left:
    st.plotly_chart(px.line(load_cohort(), x="months_since_signup", y="retention_rate",
                            color="signup_month", title="Cohort Retention"),
                    use_container_width=True)
with right:
    st.plotly_chart(px.histogram(filtered, x="risk_score", nbins=30,
                                 title="Risk Distribution"),
                    use_container_width=True)

st.subheader("Save List")
st.dataframe(
    risk[["customer_id","plan_tier","mrr","risk_score","top_risk_driver"]]
    .sort_values("risk_score", ascending=False),
    use_container_width=True,
)
st.caption(f"Test AUC: {auc:.3f}")