"""Generate the Colab notebook (TabPFN_GPU_experiments.ipynb).

Building the .ipynb from Python keeps the JSON valid and easy to review/diff.
Run:  uv run python make_colab.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "TabPFN_GPU_experiments.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n"),
    }


cells = []

cells.append(md(r"""
# TabPFN v2 on a GPU — experiments

Three experiments comparing **TabPFN v2** (the tabular foundation model from
Prior Labs) against strong baselines, on business-relevant tables.

1. **Does TabPFN match a multi-model AutoML pipeline — instantly?** TabPFN vs XGBoost vs **AutoGluon**.
2. **Where does TabPFN break?** Time-series forecasting + the *trend-extrapolation* limitation.
3. **Regression**, not just classification, on a continuous business target.

> **First:** `Runtime → Change runtime type → GPU` (T4 is enough).
> All weights are pulled from the *public* HF checkpoints, so no Prior Labs license token is needed.
"""))

cells.append(md("## 0 · Setup"))

cells.append(code(r"""
# AutoGluon is large; this cell takes a few minutes on a fresh Colab runtime.
!pip install -q tabpfn xgboost scikit-learn altair huggingface_hub pandas pyarrow
!pip install -q "autogluon.tabular[all]"   # the AutoML baseline for Experiment 1
"""))

cells.append(code(r"""
import time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "no GPU - set Runtime->GPU")

# Knobs (raise these on a bigger GPU)
MAX_CTX   = 10_000   # TabPFN in-context training rows
TEST_CAP  = 3_000    # evaluate on a fixed test sample to bound TabPFN's O(L^2) cost
AG_TIME   = 300      # AutoGluon time budget per dataset, seconds (the "instant vs ~4h AutoML" test)
SEED      = 42
"""))

cells.append(code(r"""
# Public TabPFN v2 checkpoints from the HF Hub (classifier + regressor).
from huggingface_hub import hf_hub_download
CLF_CKPT = hf_hub_download("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt")
REG_CKPT = hf_hub_download("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt")
print("downloaded:", CLF_CKPT.split('/')[-1], "+", REG_CKPT.split('/')[-1])
"""))

cells.append(md("## 1 · Data loading (straight from the Hugging Face Hub)"))

cells.append(code(r"""
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

_HF = "https://huggingface.co/datasets"
TELCO  = _HF + "/aai510-group1/telco-customer-churn/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
CREDIT = _HF + "/scikit-learn/credit-card-clients/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
GERMAN = _HF + "/marcilioduarte/german_credit_risk/resolve/main/data/processed/{f}.parquet"

# Post-churn outcome columns that leak the label; identifiers/geo with no signal.
TELCO_LEAK = ["Churn Category","Churn Reason","Churn Score","Customer Status","CLTV","Satisfaction Score"]
TELCO_DROP = ["Customer ID","City","Country","State","Lat Long","Latitude","Longitude","Zip Code","Quarter","Population"]

def encode(df):
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.5:
            out[c] = num.fillna(num.median())
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[c] = enc.fit_transform(df[[c]].astype(str)).ravel()
    return out

def load(name, include_leakage=False):
    if name == "Telco churn":
        df = pd.read_parquet(TELCO)
        y = df["Churn"].astype(int).to_numpy()
        drop = ["Churn"] + TELCO_DROP + ([] if include_leakage else TELCO_LEAK)
        X = encode(df.drop(columns=[c for c in drop if c in df.columns])).to_numpy()
        return train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    if name == "Credit-card default":
        df = pd.read_parquet(CREDIT)
        y = df["default.payment.next.month"].astype(int).to_numpy()
        X = encode(df.drop(columns=["ID","default.payment.next.month"])).to_numpy()
        return train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    if name == "German credit risk":
        Xtr = pd.read_parquet(GERMAN.format(f="x_train")).to_numpy()
        Xte = pd.read_parquet(GERMAN.format(f="x_test")).to_numpy()
        ytr = pd.read_parquet(GERMAN.format(f="y_train"))["Creditability"].astype(int).to_numpy()
        yte = pd.read_parquet(GERMAN.format(f="y_test"))["Creditability"].astype(int).to_numpy()
        return Xtr, Xte, ytr, yte
    raise ValueError(name)

DATASETS = ["Telco churn", "Credit-card default", "German credit risk"]
for d in DATASETS:
    Xtr, Xte, ytr, yte = load(d)
    print(f"{d:20s} train={Xtr.shape} test={Xte.shape} pos_rate={ytr.mean():.3f}")
"""))

cells.append(md(r"""
## 2 · Experiment 1 — does TabPFN match AutoML *instantly*?

The claim under test: *"TabPFN matches the accuracy of a four-hour AutoML pipeline — instantly."*
Here we reproduce it: **TabPFN v2** vs a tuned **XGBoost** vs **AutoGluon** (the AutoML
pipeline), reporting **ROC AUC** and **wall-clock time**. On a GPU, watch TabPFN land
in the top-left of the Pareto plot — AutoGluon-level accuracy at a fraction of the time.
"""))

cells.append(code(r"""
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from autogluon.tabular import TabularPredictor

def subsample(X, y, n):
    if n >= len(X):
        return X, y
    idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
    return X[idx], y[idx]

def run_clf(name):
    Xtr, Xte, ytr, yte = load(name)
    Xte, yte = subsample(Xte, yte, TEST_CAP)
    rows = []

    # --- XGBoost ---
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                      tree_method="hist", device=DEVICE, random_state=SEED)
    t = time.perf_counter(); m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]; dt = time.perf_counter() - t
    rows.append({"dataset": name, "model": "XGBoost", "AUC": roc_auc_score(yte, p), "time_s": dt})

    # --- TabPFN v2 ---
    Xs, ys = subsample(Xtr, ytr, MAX_CTX)
    m = TabPFNClassifier(model_path=CLF_CKPT, device=DEVICE,
                         ignore_pretraining_limits=True, random_state=SEED)
    t = time.perf_counter(); m.fit(Xs, ys)
    p = m.predict_proba(Xte)[:, 1]; dt = time.perf_counter() - t
    rows.append({"dataset": name, "model": "TabPFN v2", "AUC": roc_auc_score(yte, p), "time_s": dt})

    # --- AutoGluon (AutoML) ---
    tr = pd.DataFrame(Xtr); tr.columns = tr.columns.astype(str); tr["target"] = ytr
    te = pd.DataFrame(Xte); te.columns = te.columns.astype(str)
    t = time.perf_counter()
    pred = TabularPredictor(label="target", eval_metric="roc_auc", verbosity=0) \
        .fit(tr, time_limit=AG_TIME, presets="best_quality")
    p = pred.predict_proba(te)[1].to_numpy(); dt = time.perf_counter() - t
    rows.append({"dataset": name, "model": f"AutoGluon ({AG_TIME}s)", "AUC": roc_auc_score(yte, p), "time_s": dt})
    return rows

records = []
for d in DATASETS:
    print("running", d, "...")
    records += run_clf(d)
exp1 = pd.DataFrame(records)
exp1.round(4)
"""))

cells.append(code(r"""
import altair as alt
alt.Chart(exp1).mark_point(size=140, filled=True).encode(
    x=alt.X("time_s:Q", scale=alt.Scale(type="log"), title="train + predict time (s, log)"),
    y=alt.Y("AUC:Q", scale=alt.Scale(zero=False)),
    color="model:N", shape="model:N",
    column=alt.Column("dataset:N", title=None),
    tooltip=["dataset","model",alt.Tooltip("AUC",format=".4f"),alt.Tooltip("time_s",format=".1f")],
).properties(width=200, height=260, title="Accuracy vs. time — top-left is best")
"""))

cells.append(md(r"""
## 3 · Experiment 2 — time series & the trend-extrapolation limit

TabPFN-TS reframes forecasting as **tabular regression** on time features (this cell
does that featurization by hand). TabPFN is a strong *conditional interpolator* but,
by construction, **cannot extrapolate a trend beyond the values it saw in training** —
a structural consequence of its synthetic pretraining. We make that failure visible.
"""))

cells.append(code(r"""
from tabpfn import TabPFNRegressor

rng = np.random.RandomState(SEED)
t = np.arange(360)
trend, season = 0.06 * t, 8 * np.sin(2 * np.pi * t / 30)
y = trend + season + rng.normal(0, 1.2, t.size)

def tfeats(idx):  # time-index + Fourier seasonality features
    return np.column_stack([idx,
        np.sin(2*np.pi*idx/30), np.cos(2*np.pi*idx/30),
        np.sin(2*np.pi*idx/7),  np.cos(2*np.pi*idx/7)])

split = 260
reg = TabPFNRegressor(model_path=REG_CKPT, device=DEVICE, ignore_pretraining_limits=True)
reg.fit(tfeats(t[:split]), y[:split])
yhat = reg.predict(tfeats(t[split:]))

# The fix practitioners use: forecast the de-trended residual, then add the trend back.
from numpy.polynomial import polynomial as P
coef = P.polyfit(t[:split], y[:split], 1)
resid = y[:split] - P.polyval(t[:split], coef)
reg2 = TabPFNRegressor(model_path=REG_CKPT, device=DEVICE, ignore_pretraining_limits=True)
reg2.fit(tfeats(t[:split]), resid)
yhat_detrend = reg2.predict(tfeats(t[split:])) + P.polyval(t[split:], coef)

ts = pd.concat([
    pd.DataFrame({"t": t, "value": y, "series": "actual"}),
    pd.DataFrame({"t": t[split:], "value": yhat, "series": "TabPFN (raw)"}),
    pd.DataFrame({"t": t[split:], "value": yhat_detrend, "series": "TabPFN (de-trended)"}),
])
alt.Chart(ts).mark_line().encode(
    x="t:Q", y="value:Q", color="series:N",
).properties(width=720, height=300,
    title="Raw TabPFN flattens on the trend; de-trending restores it")
"""))

cells.append(md(r"""
## 4 · Experiment 3 — regression on a business target

Tabular foundation models do **classification *and* regression**. Here TabPFN v2's
regressor predicts a continuous business value (`Total Revenue` per telco customer)
vs an XGBoost regressor — reporting R² and MAE.
"""))

cells.append(code(r"""
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

df = pd.read_parquet(TELCO)
target = "Total Revenue"
drop = [target] + TELCO_DROP + TELCO_LEAK + ["Churn"]
Xr = encode(df.drop(columns=[c for c in drop if c in df.columns])).to_numpy()
yr = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy()
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.25, random_state=SEED)
Xte_s, yte_s = subsample(Xte, yte, TEST_CAP)

out = []
xr = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                  tree_method="hist", device=DEVICE, random_state=SEED)
xr.fit(Xtr, ytr); px = xr.predict(Xte_s)
out.append({"model":"XGBoost","R2":r2_score(yte_s,px),"MAE":mean_absolute_error(yte_s,px)})

Xs, ys = subsample(Xtr, ytr, MAX_CTX)
tr = TabPFNRegressor(model_path=REG_CKPT, device=DEVICE, ignore_pretraining_limits=True)
tr.fit(Xs, ys); pt = tr.predict(Xte_s)
out.append({"model":"TabPFN v2","R2":r2_score(yte_s,pt),"MAE":mean_absolute_error(yte_s,pt)})
pd.DataFrame(out).round(4)
"""))

cells.append(md(r"""
## 5 · Where to take it next

- **Semantic context** (the JD's exact phrase): add LLM-derived features / NL row
  serializations and test whether AUC improves — probing the *tables + language* frontier.
- **TabArena / `inria-soda/tabular-benchmark`**: reproduce a slice of the public leaderboard.
- **Relational foundation models**: extend from single-table to multi-table
  (relational) prediction, where business semantics live across joined tables.
- For real forecasting use the **`tabpfn-time-series`** package (automates the
  featurization in Experiment 2 and adds covariate support).
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
