"""Precompute every (dataset x model x context-size) result once and cache to
results.json so the marimo app is instant and reproducible for a live demo.

Run:  uv run python -m precompute
"""
from __future__ import annotations

import json
import os
import time

import bench

OUT = os.path.join(os.path.dirname(__file__), "public", "results.json")


def _save(results: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {
        "datasets": bench.DATASETS,
        "test_cap": bench.TEST_CAP,
        "tabpfn_max_context": bench.TABPFN_MAX_CONTEXT,
        "results": results,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)


def main() -> None:
    results: list[dict] = []
    t_start = time.perf_counter()

    for ds in bench.DATASETS:
        grid = bench.context_grid(ds)
        leak_opts = [False, True] if bench.DATASETS[ds]["supports_leakage_toggle"] else [False]
        for leak in leak_opts:
            for model in ("XGBoost", "TabPFN v2"):
                for n in grid:
                    t0 = time.perf_counter()
                    r = bench.run_model(ds, model, n, include_leakage=leak)
                    r["include_leakage"] = leak
                    results.append(r)
                    _save(results)  # incremental: app is usable mid-run
                    print(
                        f"[{time.perf_counter()-t_start:6.1f}s] {ds:20s} "
                        f"{model:10s} leak={int(leak)} n={n:5d} "
                        f"AUC={r['auc']:.4f} ({time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )

    print(f"\nWrote {len(results)} results -> {OUT} in {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
