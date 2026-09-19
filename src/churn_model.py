import logging
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .db import load_features

log = logging.getLogger(__name__)
NUM = ["tenure_months","avg_monthly_logins","total_tickets","pct_tickets_unresolved"]
CAT = ["plan_tier"]

def train_and_score(path=None, seed=42):
    df = load_features(path) if path else load_features()
    x_train, x_test, y_train, y_test = train_test_split(
        df[NUM+CAT], df.churned, test_size=.2, stratify=df.churned, random_state=seed
    )
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=2000, random_state=seed))
    ])
    pipe.fit(x_train, y_train)
    auc = float(roc_auc_score(y_test, pipe.predict_proba(x_test)[:,1]))
    log.info("AUC %.4f rows=%s", auc, len(df))

    active = df[df.churned==0].copy()
    active["risk_score"] = pipe.predict_proba(active[NUM+CAT])[:,1]
    z = pipe.named_steps["pre"].transform(active[NUM+CAT])
    z = z.toarray() if hasattr(z, "toarray") else z
    contribution = z * pipe.named_steps["model"].coef_[0]
    names = pipe.named_steps["pre"].get_feature_names_out()
    active["top_risk_driver"] = [
        names[i].replace("num__","").replace("cat__","")
        for i in np.argmax(contribution, axis=1)
    ]
    return active, auc