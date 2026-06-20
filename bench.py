"""Benchmark core: load business-relevant tabular datasets from the HF Hub and
compare TabPFN v2 (a tabular foundation model) against XGBoost.

Datasets map to common enterprise use cases for tabular foundation models:
  - Telco churn        -> customer churn
  - Credit-card default-> payment-default / payment-delay risk
  - German credit risk -> credit / counterparty risk

All data is pulled straight from the Hugging Face Hub's auto-converted parquet.
"""
from __future__ import annotations

import functools
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

SEED = 42
# TabPFN v2 is designed for small data; cap the in-context training set so CPU
# inference stays fast enough for a live demo.
TABPFN_MAX_CONTEXT = 2_000
# Bound TabPFN's O(L^2) inference cost by evaluating on a fixed test sample.
TEST_CAP = 1_500
# Smaller ensemble keeps CPU inference snappy with negligible accuracy loss.
TABPFN_N_ESTIMATORS = 2

_HF = "https://huggingface.co/datasets"
_TELCO = (
    f"{_HF}/aai510-group1/telco-customer-churn/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)
_CREDIT = (
    f"{_HF}/scikit-learn/credit-card-clients/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
_GERMAN = _HF + "/marcilioduarte/german_credit_risk/resolve/main/data/processed/{f}.parquet"

# Columns that leak the label in the telco dataset (post-churn outcome fields).
TELCO_LEAKAGE = [
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Customer Status",
    "CLTV",
    "Satisfaction Score",
]
# Identifiers / geo / constant columns with no predictive signal.
TELCO_DROP = [
    "Customer ID", "City", "Country", "State", "Lat Long",
    "Latitude", "Longitude", "Zip Code", "Quarter", "Population",
]

DATASETS = {
    "Telco churn": {
        "use_case": "customer churn",
        "supports_leakage_toggle": True,
    },
    "Credit-card default": {
        "use_case": "payment-default risk",
        "supports_leakage_toggle": False,
    },
    "German credit risk": {
        "use_case": "credit / counterparty risk",
        "supports_leakage_toggle": False,
    },
}


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to a fully-numeric frame: numeric where possible, else ordinal-encode."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.5:
            out[col] = numeric.fillna(numeric.median())
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[col] = enc.fit_transform(df[[col]].astype(str)).ravel()
    return out


@functools.lru_cache(maxsize=8)
def load_dataset(name: str, include_leakage: bool = False):
    """Return (X_train, y_train, X_test, y_test) as numpy arrays, plus pos label rate."""
    if name == "Telco churn":
        df = pd.read_parquet(_TELCO.format(split="train"))
        y = df["Churn"].astype(int).to_numpy()
        drop = ["Churn"] + TELCO_DROP + ([] if include_leakage else TELCO_LEAKAGE)
        X = _encode(df.drop(columns=[c for c in drop if c in df.columns]))
        Xtr, Xte, ytr, yte = train_test_split(
            X.to_numpy(), y, test_size=0.25, random_state=SEED, stratify=y
        )
    elif name == "Credit-card default":
        df = pd.read_parquet(_CREDIT)
        y = df["default.payment.next.month"].astype(int).to_numpy()
        X = _encode(df.drop(columns=["ID", "default.payment.next.month"]))
        Xtr, Xte, ytr, yte = train_test_split(
            X.to_numpy(), y, test_size=0.25, random_state=SEED, stratify=y
        )
    elif name == "German credit risk":
        Xtr = pd.read_parquet(_GERMAN.format(f="x_train")).to_numpy()
        Xte = pd.read_parquet(_GERMAN.format(f="x_test")).to_numpy()
        ytr = pd.read_parquet(_GERMAN.format(f="y_train"))["Creditability"].astype(int).to_numpy()
        yte = pd.read_parquet(_GERMAN.format(f="y_test"))["Creditability"].astype(int).to_numpy()
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return Xtr, ytr, Xte, yte, float(np.mean(ytr))


def _build(model_name: str):
    if model_name == "TabPFN v2":
        from tabpfn import TabPFNClassifier

        # Use the publicly-downloaded HF checkpoint directly so we don't need a
        # Prior Labs license token for local inference.
        ckpt = os.path.join(
            os.path.dirname(__file__), "weights", "tabpfn-v2-classifier.ckpt"
        )
        return TabPFNClassifier(
            model_path=ckpt,
            device="cpu",
            ignore_pretraining_limits=True,
            n_estimators=TABPFN_N_ESTIMATORS,
            random_state=SEED,
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=4,
            random_state=SEED,
        )
    raise ValueError(model_name)


@functools.lru_cache(maxsize=256)
def run_model(name: str, model_name: str, n_context: int, include_leakage: bool = False):
    """Train one model on `n_context` rows and evaluate. Returns a metrics dict."""
    Xtr, ytr, Xte, yte, _ = load_dataset(name, include_leakage)

    n = min(n_context, len(Xtr))
    if model_name == "TabPFN v2":
        n = min(n, TABPFN_MAX_CONTEXT)
    if n < len(Xtr):
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(Xtr), size=n, replace=False)
        Xs, ys = Xtr[idx], ytr[idx]
    else:
        Xs, ys = Xtr, ytr

    if len(Xte) > TEST_CAP:
        rng = np.random.RandomState(SEED)
        tidx = rng.choice(len(Xte), size=TEST_CAP, replace=False)
        Xte, yte = Xte[tidx], yte[tidx]

    model = _build(model_name)
    t0 = time.perf_counter()
    model.fit(Xs, ys)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict_proba(Xte)[:, 1]
    pred_t = time.perf_counter() - t0

    fpr, tpr, _ = roc_curve(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    # Reliability diagram (manual, fixed bins so curves are comparable).
    bins = np.linspace(0, 1, 11)
    which = np.digitize(proba, bins) - 1
    cal_pred, cal_true = [], []
    for b in range(10):
        m = which == b
        if m.sum() > 0:
            cal_pred.append(float(proba[m].mean()))
            cal_true.append(float(yte[m].mean()))

    return {
        "dataset": name,
        "model": model_name,
        "n_context": int(n),
        "auc": float(roc_auc_score(yte, proba)),
        "accuracy": float(accuracy_score(yte, (proba >= 0.5).astype(int))),
        "avg_precision": float(average_precision_score(yte, proba)),
        "fit_time": fit_t,
        "predict_time": pred_t,
        "total_time": fit_t + pred_t,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "calibration": {"pred": cal_pred, "true": cal_true},
    }


def context_grid(name: str) -> list[int]:
    Xtr, *_ = load_dataset(name)
    n = len(Xtr)
    grid = [g for g in (100, 250, 500, 1000, 2000) if g < n]
    grid.append(min(n, TABPFN_MAX_CONTEXT))
    return sorted(set(grid))


if __name__ == "__main__":
    # Smoke test: prove real numbers come out before wiring the UI.
    for ds in DATASETS:
        Xtr, ytr, Xte, yte, rate = load_dataset(ds)
        print(f"\n{ds}: train={Xtr.shape} test={Xte.shape} pos_rate={rate:.3f}")
        for mdl in ("TabPFN v2", "XGBoost"):
            r = run_model(ds, mdl, n_context=1000)
            print(
                f"  {mdl:10s} AUC={r['auc']:.4f} acc={r['accuracy']:.4f} "
                f"AP={r['avg_precision']:.4f} time={r['total_time']:.2f}s "
                f"(n={r['n_context']})"
            )
