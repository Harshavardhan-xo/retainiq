import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="RetainIQ | Customer Success Command Center", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:1.2rem 1.4rem;border-radius:18px;background:linear-gradient(135deg,#0b1f33,#163d59);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;color:#cfe7f5}
.badge{display:inline-block;background:#0ea5a5;color:white;padding:.25rem .6rem;border-radius:999px;font-size:.75rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate():
    rng=np.random.default_rng(42); n=15000
    plans=["Starter","Growth","Scale","Enterprise"]
    regions=["North America","UKI","DACH","India","ANZ","SEA"]
    industries=["SaaS","Fintech","Retail","Healthcare","Manufacturing","Professional Services"]
    segments=["SMB","Mid-Market","Enterprise"]
    signup=pd.Timestamp("2024-01-01")+pd.to_timedelta(rng.integers(0,930,n),unit="D")
    plan=rng.choice(plans,n,p=[.35,.34,.20,.11]); region=rng.choice(regions,n)
    industry=rng.choice(industries,n); segment=rng.choice(segments,n,p=[.48,.38,.14])
    seats=np.maximum(2,rng.lognormal(3,.65,n).astype(int))
    mrr=np.array([{"Starter":89,"Growth":349,"Scale":1299,"Enterprise":4200}[p] for p in plan])*rng.lognormal(0,.22,n)*(1+.12*np.log1p(seats))
    adoption=np.clip(rng.beta(4,2,n)*100,2,99); logins=np.maximum(0,rng.normal(18+adoption*.20,8,n))
    tickets=rng.poisson(np.maximum(1.2,4.8-adoption*.025),n)
    unresolved=np.clip(rng.beta(1.8,8,n)+tickets*.01,0,1)
    nps=np.clip(rng.normal(42+adoption*.25-tickets*.7,18,n),-80,90)
    expansion=np.clip(.25+.006*adoption+.05*(plan=="Enterprise")-.03*tickets+rng.normal(0,.08,n),0,1)
    logit=-2.3-.045*adoption+.06*tickets+1.15*unresolved-.025*np.minimum(logins,40)+.45*(plan=="Starter")-.35*expansion+rng.normal(0,.38,n)
    churn= rng.random(n) < 1/(1+np.exp(-logit))
    customers=pd.DataFrame({"customer_id":np.arange(1,n+1),"signup_date":signup.dt.date,"region":region,"plan_tier":plan,
                            "industry":industry,"segment":segment,"seats":seats,"mrr":mrr.round(2),"adoption_score":adoption.round(1),
                            "logins_30d":logins.round(1),"tickets_30d":tickets,"unresolved_pct":unresolved.round(3),
                            "nps":nps.round(0),"expansion_propensity":expansion.round(3),"churned":churn})
    months=pd.date_range(pd.Timestamp.today().normalize()-pd.DateOffset(months=11),periods=12,freq="MS")
    snap=pd.DataFrame({"snapshot_month":np.repeat(months,n),"customer_id":np.tile(customers.customer_id.values,12)})
    cidx=snap.customer_id.to_numpy()-1; midx=np.repeat(np.arange(12),n); season=1+.04*np.sin(midx/11*2*np.pi)
    snap["active"]=~customers.iloc[cidx].churned.to_numpy(); snap["logins"]=np.maximum(0,customers.iloc[cidx].logins_30d.to_numpy()*season+rng.normal(0,2,len(snap)))
    cohort=customers.groupby(pd.to_datetime(customers.signup_date).dt.to_period("M")).agg(customers=("customer_id","count"),active=("churned",lambda s:(~s).sum())).reset_index()
    cohort["cohort_month"]=cohort.signup_date.astype(str); cohort["retention"]=cohort.active/cohort.customers
    return customers,snap,cohort

@st.cache_resource
def train_model(df):
    num=["adoption_score","logins_30d","tickets_30d","unresolved_pct","nps","expansion_propensity"]; cat=["plan_tier","region","industry","segment"]
    X=df[num+cat]; y=df.churned.astype(int)
    a,b,ya,yb=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    pipe=Pipeline([("pre",pre),("model",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=42))])
    pipe.fit(a,ya); auc=float(roc_auc_score(yb,pipe.predict_proba(b)[:,1]))
    active=df[~df.churned].copy(); active["risk_score"]=pipe.predict_proba(active[num+cat])[:,1]
    return active,auc

customers,snapshots,cohort=generate(); active,auc=train_model(customers)
with st.sidebar:
    st.header("Portfolio Controls")
    threshold=st.slider("Risk threshold",.25,.90,.55,.05)
    regions=st.multiselect("Region",sorted(customers.region.unique()),default=sorted(customers.region.unique()))
    plans=st.multiselect("Plan",sorted(customers.plan_tier.unique()),default=sorted(customers.plan_tier.unique()))
    segments=st.multiselect("Segment",sorted(customers.segment.unique()),default=sorted(customers.segment.unique()))
    min_mrr=st.slider("Minimum MRR",0,5000,0,100)

active_f=active[active.region.isin(regions)&active.plan_tier.isin(plans)&active.segment.isin(segments)&(active.mrr>=min_mrr)]
risk=active_f[active_f.risk_score>=threshold]

st.markdown('<div class="hero"><span class="badge">CUSTOMER SUCCESS • REVOPS</span><h1>RetainIQ — Customer Retention Command Center</h1><p>Account-level churn risk, revenue exposure, adoption health and expansion signals across a 15,000-customer synthetic SaaS portfolio.</p></div>',unsafe_allow_html=True)

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Active ARR",f"USD {active.mrr.sum()*12/1e6:.2f}M")
c2.metric("MRR at Risk",f"USD {risk.mrr.sum()/1e3:.1f}K")
c3.metric("Logo Churn",f"{customers.churned.mean():.1%}")
c4.metric("At-Risk Accounts",f"{len(risk):,}")
c5.metric("Model AUC",f"{auc:.3f}")
st.caption(f"Data scale: {len(customers):,} customers • {len(snapshots):,} monthly activity rows • {customers.tickets_30d.sum():,} support-ticket observations")

t1,t2,t3,t4=st.tabs(["Executive Overview","Retention & Cohorts","Risk Explorer","Account Workbench"])
with t1:
    l,r=st.columns(2)
    seg=customers.groupby("segment",as_index=False).agg(mrr=("mrr","sum"),churn=("churned","mean"))
    l.plotly_chart(px.bar(seg,x="segment",y="mrr",title="MRR by Segment",text_auto=".3s"),use_container_width=True)
    r.plotly_chart(px.scatter(customers,x="adoption_score",y="mrr",size="seats",color="plan_tier",hover_data=["customer_id","industry","region"],title="Adoption vs MRR"),use_container_width=True)
    q=customers.groupby("industry",as_index=False).agg(accounts=("customer_id","count"),mrr=("mrr","sum"),churn=("churned","mean"))
    q["risk_mrr_proxy"]=q.mrr*q.churn
    st.dataframe(q.sort_values("risk_mrr_proxy",ascending=False),use_container_width=True,hide_index=True)
with t2:
    l,r=st.columns(2)
    l.plotly_chart(px.line(cohort.sort_values("cohort_month"),x="cohort_month",y="retention",markers=True,title="Signup Cohort Retention"),use_container_width=True)
    plan_stats=customers.groupby("plan_tier",as_index=False).churned.mean()
    r.plotly_chart(px.bar(plan_stats,x="plan_tier",y="churned",title="Churn by Plan"),use_container_width=True)
    monthly=snapshots.groupby("snapshot_month",as_index=False).agg(active_accounts=("active","sum"),avg_logins=("logins","mean"))
    st.plotly_chart(px.line(monthly,x="snapshot_month",y=["active_accounts","avg_logins"],title="Portfolio Health Trend"),use_container_width=True)
with t3:
    l,r=st.columns(2); sample=active.sample(min(8000,len(active)),random_state=42)
    l.plotly_chart(px.scatter(sample,x="adoption_score",y="risk_score",color="plan_tier",size=np.clip(sample.mrr,50,2500),hover_data=["customer_id","industry","region"],title="Risk vs Adoption"),use_container_width=True)
    r.plotly_chart(px.histogram(active,x="risk_score",color="segment",nbins=30,barmode="overlay",title="Risk Distribution"),use_container_width=True)
    drivers=pd.DataFrame({"driver":["Low adoption","High unresolved tickets","Low login frequency","Weak NPS","Starter tier"],"impact":[32,24,20,14,10]})
    st.plotly_chart(px.bar(drivers,x="impact",y="driver",orientation="h",title="Illustrative Risk Driver Mix"),use_container_width=True)
with t4:
    view=risk[["customer_id","region","plan_tier","segment","industry","mrr","adoption_score","logins_30d","tickets_30d","unresolved_pct","nps","expansion_propensity","risk_score"]].sort_values("risk_score",ascending=False)
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download Prioritized Save List",view.to_csv(index=False).encode(),"retainiq_save_list.csv","text/csv")
