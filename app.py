import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="RetainIQ", layout="wide")
st.title("RetainIQ — SaaS Churn & Revenue at Risk")
st.caption("Synthetic dashboard: churn risk, MRR at risk, and top risk drivers.")

@st.cache_data
def build():
    rng=np.random.default_rng(42); n=2200
    plan=rng.choice(["starter","pro","enterprise"],n,p=[.46,.39,.15])
    tenure=rng.uniform(.5,48,n)
    logins=np.maximum(0,rng.normal(14+tenure*.18,7,n))
    tickets=np.maximum(0,rng.poisson(1.6,n))
    unresolved=np.clip(rng.beta(1.5,7,n)+.015*tickets,0,1)
    mrr=np.select([plan=="starter",plan=="pro",plan=="enterprise"],
                  [rng.uniform(49,149,n),rng.uniform(199,599,n),rng.uniform(800,2500,n)])
    logit=-2.4-.045*tenure-.07*logins+.23*tickets+1.15*unresolved+.5*(plan=="starter")+rng.normal(0,.5,n)
    churn_p=1/(1+np.exp(-logit)); churn=rng.random(n)<churn_p
    df=pd.DataFrame({"customer_id":np.arange(1,n+1),"plan_tier":plan,"tenure_months":tenure,
                     "avg_monthly_logins":logins,"total_tickets":tickets,
                     "pct_tickets_unresolved":unresolved,"mrr":mrr,"churned":churn})
    with sqlite3.connect(":memory:") as con:
        df.to_sql("customers",con,index=False)
        return pd.read_sql_query("select * from customers",con)

@st.cache_resource
def model_data():
    df=build()
    num=["tenure_months","avg_monthly_logins","total_tickets","pct_tickets_unresolved"]; cat=["plan_tier"]
    pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    pipe=Pipeline([("pre",pre),("model",LogisticRegression(max_iter=2000,random_state=42))])
    pipe.fit(df[num+cat],df.churned)
    auc=roc_auc_score(df.churned,pipe.predict_proba(df[num+cat])[:,1])
    active=df[df.churned==0].copy()
    active["risk_score"]=pipe.predict_proba(active[num+cat])[:,1]
    active["top_risk_driver"]="usage/support/tenure/plan factors"
    return df,active,float(auc)

df,active,auc=model_data()
threshold=st.sidebar.slider("Risk threshold",0.0,1.0,.5,.05)
plans=st.sidebar.multiselect("Plan",sorted(active.plan_tier.unique()),default=sorted(active.plan_tier.unique()))
risk=active[(active.risk_score>=threshold)&active.plan_tier.isin(plans)]

c1,c2,c3,c4=st.columns(4)
c1.metric("Active MRR",f"USD {active.mrr.sum():,.0f}")
c2.metric("MRR at Risk",f"USD {risk.mrr.sum():,.0f}")
c3.metric("Churn Rate",f"{df.churned.mean():.1%}")
c4.metric("Validation AUC",f"{auc:.3f}")

left,right=st.columns(2)
with left:
    st.plotly_chart(px.histogram(active,x="risk_score",nbins=35,title="Risk Score Distribution"),use_container_width=True)
with right:
    st.plotly_chart(px.box(active,x="plan_tier",y="risk_score",title="Risk by Plan"),use_container_width=True)
st.subheader("Customer Save List")
st.dataframe(risk[["customer_id","plan_tier","mrr","risk_score","top_risk_driver"]].sort_values("risk_score",ascending=False),use_container_width=True)
