# app.py
import numpy as np, pandas as pd, duckdb
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# Safety guard so running `python app.py` gives a clear message
if not st.runtime.exists():
    raise SystemExit("This file is a Streamlit app. Run it with:  python -m streamlit run app.py")

st.set_page_config(page_title="Pruning Playground — Adult Income", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.rename(columns={"class":"target"})
    fe = duckdb.sql("""
        WITH base AS (
            SELECT *,
                CASE WHEN education IN ('Bachelors','Masters','Doctorate') THEN 1 ELSE 0 END AS is_degree,
                CASE WHEN "hours-per-week" >= 50 THEN 1 ELSE 0 END AS long_hours,
                CASE WHEN "marital-status" LIKE 'Married-%' THEN 1 ELSE 0 END AS married_any,
                CASE 
                  WHEN "capital-gain"=0 THEN '0'
                  WHEN "capital-gain"<2000 THEN '0-2k'
                  WHEN "capital-gain"<5000 THEN '2-5k'
                  WHEN "capital-gain"<10000 THEN '5-10k'
                  ELSE '10k+'
                END AS cap_gain_bucket
            FROM df
        )
        SELECT * FROM base
    """).df()
    return fe

df = load_data()
X = df.drop(columns=["target"])
y = (df["target"]==">50K").astype(int)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

st.title("Pruning Playground — Decision Tree on Adult Income")
st.write("Tune **pre-pruning** and **post-pruning** to see how complexity vs performance trades off.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    max_depth = st.slider("max_depth (None=0)", 0, 30, 8)
with col2:
    min_leaf = st.slider("min_samples_leaf", 1, 100, 10)
with col3:
    min_split = st.slider("min_samples_split", 2, 200, 20)
with col4:
    ccp_alpha = st.slider("ccp_alpha (post-pruning)", 0.0, 0.02, 0.0, 0.0005)

clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=None if max_depth==0 else max_depth,
    min_samples_leaf=min_leaf,
    min_samples_split=min_split,
    ccp_alpha=ccp_alpha
)
pipe = Pipeline([("prep", preprocess), ("clf", clf)])
pipe.fit(X, y)

proba = pipe.predict_proba(X)[:,1]
auc = roc_auc_score(y, proba)
st.metric("AUC", f"{auc:.3f}")

fpr, tpr, _ = roc_curve(y, proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
st.plotly_chart(fig, use_container_width=True)

thr = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.01)
pred = (proba >= thr).astype(int)
cm = confusion_matrix(y, pred)

fig2 = go.Figure(data=go.Heatmap(
    z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
    text=cm, texttemplate="%{text}"
))
fig2.update_layout(title=f"Confusion Matrix @ threshold={thr:.2f}")
st.plotly_chart(fig2, use_container_width=True)
