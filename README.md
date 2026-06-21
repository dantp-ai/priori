# priori

Tabular foundation models predict *a priori* — in-context, with no training.
`priori` is an interactive [marimo](https://marimo.io) benchmark of TabPFN v2
against tuned XGBoost and [AutoGluon](https://auto.gluon.ai/stable/index.html) on real business tables, charting where a model's blank-slate prior wins on small data — and where gradient-boosted trees catch up.

TabPFN v2 is a tabular foundation model from [Prior Labs](https://priorlabs.ai/). The three Hugging Face
datasets map to common enterprise use cases for it:

| Dataset (HF) | Use case | Rows |
|---|---|---|
| `aai510-group1/telco-customer-churn` | customer churn | ~5.6k |
| `scikit-learn/credit-card-clients` | payment-default risk | 30k |
| `marcilioduarte/german_credit_risk` | credit / counterparty risk | 1k |

## What it shows

- **Data efficiency**: On telco churn at **100 rows**, TabPFN
  scores AUC ~0.86 vs XGBoost ~0.77; XGBoost only catches up near full data.
- **Accuracy vs. time (Pareto)**: TabPFN buys accuracy with compute. *Note:*
  timing in the dashboard is **CPU**; TabPFN's "instant" claim is a GPU result
  (reproduce it with the GPU notebook below).
- **Calibration & PR curves**: probability quality matters for business risk
  scoring, and churn is imbalanced (~27% positive), so PR > accuracy.
- **Leakage toggle**: flipping in the post-churn outcome columns
  (`Churn Score`, `Customer Status`, …) sends AUC to ~0.99.

## Setup

```bash
uv sync                      # creates .venv (Python 3.12+) and installs deps
source .venv/bin/activate

# TabPFN v8 gates weights behind a license token, but the HF checkpoint is public:
hf download Prior-Labs/TabPFN-v2-clf tabpfn-v2-classifier.ckpt --local-dir weights
hf download Prior-Labs/TabPFN-v2-reg tabpfn-v2-regressor.ckpt  --local-dir weights
```

## Run

```bash
# macOS: avoid an OpenMP/torch/XGBoost segfault
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false

python precompute.py                       # build public/results.json (real runs, ~3 min CPU)
marimo run app.py                          # open the reactive dashboard
marimo export html-wasm app.py -o build --mode run && \
  python -m http.server --directory build  # preview the shareable static app
```

## GPU experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dantp-ai/priori/blob/main/TabPFN_GPU_experiments.ipynb)

`TabPFN_GPU_experiments.ipynb` tests the *"matches a 4-hour AutoML pipeline,
instantly"* claim (TabPFN vs XGBoost vs **AutoGluon**), shows TabPFN's
trend-extrapolation limit on time series, and runs regression on a continuous
business target.

**Run it on a free GPU — no install, no clone:**

1. Click the **Open in Colab** badge above. Colab loads the notebook straight
   from this public GitHub repo; no authorization needed.
2. In Colab: **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.** Weights download from the public HF checkpoints, so no
   token is required.


## Files

- `bench.py` — data loading/cleaning + `run_model()`.
- `precompute.py` — runs the full grid once → `public/results.json`.
- `app.py` — reactive marimo dashboard (Altair charts).
- `make_colab.py` — regenerates the GPU notebook.
