"""TabPFN v2 vs XGBoost on business tables — a reactive marimo app.

Run:    marimo run app.py
Edit:   marimo edit app.py

Reads the precomputed results.json (see precompute.py). All numbers are real
runs of TabPFN v2 (HF weights, local CPU inference) vs a tuned XGBoost.
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pd


@app.cell
def _(Path, json, mo, pd):
    # mo.notebook_location() resolves to a local dir when run locally and to the
    # served URL when exported to WASM — so reading from public/ works in both.
    src = str(mo.notebook_location() / "public" / "results.json")
    if src.startswith(("http://", "https://")):
        import urllib.request

        raw = urllib.request.urlopen(src).read()
    else:
        raw = Path(src).read_bytes()
    payload = json.loads(raw)
    res = payload["results"]

    # Flat metrics table for scatter / line / leaderboard charts.
    metrics = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "n_context": r["n_context"],
                "leakage": r["include_leakage"],
                "AUC": r["auc"],
                "accuracy": r["accuracy"],
                "avg_precision": r["avg_precision"],
                "total_time": r["total_time"],
            }
            for r in res
        ]
    )
    # Curves keyed for the selected configuration.
    curves = {
        (r["dataset"], r["model"], r["n_context"], r["include_leakage"]): r
        for r in res
    }
    use_cases = {k: v["use_case"] for k, v in payload["datasets"].items()}
    return curves, metrics, payload, use_cases


@app.cell
def _(mo, payload, use_cases):
    mo.md(
        f"""
        # 🧮 Tabular Foundation Models on Business Data
        ### TabPFN v2 (a tabular foundation model from Prior Labs) vs. XGBoost

        Three Hugging Face datasets mapped to common enterprise use cases for
        tabular foundation models — **churn**, **payment-default risk**, **credit risk**.
        Every number below is a *real* run: TabPFN v2 with the public HF
        checkpoint doing local **CPU** in-context inference, against a tuned
        XGBoost. Evaluated on a fixed {payload['test_cap']}-row test sample;
        TabPFN context capped at {payload['tabpfn_max_context']} rows.

        > **Note:** TabPFN's "instant" claim is a *GPU*
        > result. On CPU it is much slower than XGBoost — the interesting story
        > is the **accuracy-per-row** and **calibration**, not wall-clock here.
        """
    )
    return


@app.cell
def _(metrics, mo):
    dataset = mo.ui.dropdown(
        options=list(metrics["dataset"].unique()),
        value="Telco churn",
        label="**Dataset**",
    )
    leak = mo.ui.switch(value=False, label="Include leakage columns (Telco only)")
    return dataset, leak


@app.cell
def _(dataset, leak, metrics, mo):
    grid = sorted(
        metrics[
            (metrics["dataset"] == dataset.value)
            & (metrics["leakage"] == (leak.value and dataset.value == "Telco churn"))
        ]["n_context"].unique()
    )
    context = mo.ui.dropdown(
        options={str(g): g for g in grid},
        value=str(grid[-1]),
        label="**In-context training rows**",
    )
    mo.hstack([dataset, context, leak], justify="start", gap=2)
    return context, grid


@app.cell
def _(dataset, leak):
    # Leakage toggle only applies to the dataset that has leakage columns.
    leak_on = leak.value and dataset.value == "Telco churn"
    return (leak_on,)


@app.cell
def _(alt, context, dataset, leak_on, metrics, mo):
    d = metrics[(metrics["dataset"] == dataset.value) & (metrics["leakage"] == leak_on)]

    line = (
        alt.Chart(d)
        .mark_line(point=True, opacity=0.85)
        .encode(
            x=alt.X("total_time:Q", scale=alt.Scale(type="log"), title="train + predict time (s, log)"),
            y=alt.Y("AUC:Q", scale=alt.Scale(zero=False), title="ROC AUC"),
            color=alt.Color("model:N", title="model"),
            tooltip=["model", "n_context", alt.Tooltip("AUC", format=".4f"), alt.Tooltip("total_time", format=".2f")],
        )
    )
    # Ring the points at the selected in-context size so the dropdown visibly
    # drives this chart too (the line still shows every context size).
    selected = d[d["n_context"] == context.value]
    highlight = (
        alt.Chart(selected)
        .mark_point(size=260, filled=False, stroke="black", strokeWidth=2)
        .encode(x="total_time:Q", y="AUC:Q")
    )
    hero = (line + highlight).properties(
        height=320, title="Accuracy vs. time — each point is a context size"
    )
    mo.md(
        "## 🎯 Accuracy vs. time (the Pareto picture)\n"
        "Top-left = better. TabPFN buys accuracy with compute; XGBoost is cheap. "
        "The ringed points mark the selected in-context size."
    )
    return d, hero


@app.cell
def _(hero, mo):
    mo.ui.altair_chart(hero)
    return


@app.cell
def _(alt, d, mo):
    sweep = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_context:Q", scale=alt.Scale(type="log"), title="in-context training rows (log)"),
            y=alt.Y("AUC:Q", scale=alt.Scale(zero=False)),
            color="model:N",
            tooltip=["model", "n_context", alt.Tooltip("AUC", format=".4f")],
        )
        .properties(height=300, title="Data efficiency: AUC vs. number of training rows")
    )
    mo.md("## 🎚️ Data efficiency\nHow quickly does each model reach its ceiling as rows grow?")
    return (sweep,)


@app.cell
def _(mo, sweep):
    mo.ui.altair_chart(sweep)
    return


@app.cell
def _(alt, context, curves, dataset, leak_on, mo, pd):
    roc_rows, pr_rows, cal_rows = [], [], []
    for model in ("XGBoost", "TabPFN v2"):
        r = curves.get((dataset.value, model, context.value, leak_on))
        if not r:
            continue
        roc_rows += [{"model": model, "fpr": f, "tpr": t} for f, t in zip(r["roc"]["fpr"], r["roc"]["tpr"])]
        pr_rows += [{"model": model, "recall": rc, "precision": p} for rc, p in zip(r["pr"]["recall"], r["pr"]["precision"])]
        cal_rows += [{"model": model, "pred": pp, "true": tt} for pp, tt in zip(r["calibration"]["pred"], r["calibration"]["true"])]

    roc = (
        alt.Chart(pd.DataFrame(roc_rows))
        .mark_line()
        .encode(x=alt.X("fpr:Q", title="false positive rate"), y=alt.Y("tpr:Q", title="true positive rate"), color="model:N")
        .properties(height=260, title="ROC")
    )
    pr = (
        alt.Chart(pd.DataFrame(pr_rows))
        .mark_line()
        .encode(x=alt.X("recall:Q"), y=alt.Y("precision:Q", scale=alt.Scale(zero=False)), color="model:N")
        .properties(height=260, title="Precision–Recall (imbalance-aware)")
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
    cal = (
        alt.Chart(pd.DataFrame(cal_rows)).mark_line(point=True).encode(
            x=alt.X("pred:Q", title="predicted probability"),
            y=alt.Y("true:Q", title="observed frequency"),
            color="model:N",
        )
        + diag
    ).properties(height=260, title="Calibration (closer to diagonal = better)")

    mo.md(f"## 📉 Diagnostics at **{context.value}** rows")
    return cal, pr, roc


@app.cell
def _(cal, mo, pr, roc):
    mo.hstack([roc, pr, cal], widths="equal")
    return


@app.cell
def _(metrics, mo):
    # Leaderboard at each dataset's largest context (no-leakage view).
    base = metrics[~metrics["leakage"]]
    idx = base.groupby(["dataset", "model"])["n_context"].idxmax()
    board = (
        base.loc[idx, ["dataset", "model", "n_context", "AUC", "avg_precision", "accuracy", "total_time"]]
        .sort_values(["dataset", "AUC"], ascending=[True, False])
        .round(4)
        .reset_index(drop=True)
    )
    mo.md("## 🏆 Leaderboard (largest context per dataset)")
    return (board,)


@app.cell
def _(board, mo):
    mo.ui.table(board, selection=None, pagination=False)
    return


if __name__ == "__main__":
    app.run()
