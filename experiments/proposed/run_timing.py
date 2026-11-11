# Calisma suresi: keskin karar vs bulanik katman (test asamasi, ornek basina)
# yonerge 6.2'deki "Execution Time" satiri icin
# calistirma: python -m experiments.proposed.run_timing
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_botiot, load_skin, split_train_test
from src.fuzzy_detect import FuzzyConfig, FuzzyDetector
from src.naim import NaIM, NaIMConfig

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_RUNS = 5


def timed(f, reps=3):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


rows = []
for ds, frac in ((load_botiot("clean"), 0.5), (load_skin(), 0.02)):
    for run_id in range(N_RUNS):
        rng = np.random.default_rng(1200 + run_id)
        S, N, X_te, y_te = split_train_test(ds, frac, rng)
        X_te = X_te[:20000]
        t = time.perf_counter()
        model = NaIM(NaIMConfig()).fit(S, N, rng)
        t_train = time.perf_counter() - t
        n = len(X_te)
        row = {"data": ds.name, "run": run_id, "n_test": n, "NaIM egitim (s)": t_train,
               "Keskin (us/ornek)": 1e6 * timed(lambda: model.predict(X_te)) / n}
        for name, cfg in (("Mamdani", FuzzyConfig("tri", "mamdani")), ("Sugeno", FuzzyConfig("gauss", "sugeno"))):
            fd = FuzzyDetector.from_model(model, N, cfg)
            row[f"FIS {name} (us/ornek)"] = 1e6 * timed(lambda: fd.score(X_te)) / n
        g1, g2 = fd.inputs(X_te)
        fd.cfg = FuzzyConfig("tri", "mamdani")
        row["sadece cikarim, Mamdani (us/ornek)"] = 1e6 * timed(lambda: fd.score_from_inputs(g1, g2)) / n
        rows.append(row)
    print("tamam:", ds.name)

df = pd.DataFrame(rows)
g = df.drop(columns="run").groupby("data", sort=False).mean().round(2)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "timing_runs.csv", index=False)
(OUT / "timing.md").write_text("## Calisma suresi (5 tekrar ortalamasi, en iyi 3 olcum)\n\n" + g.to_markdown() + "\n",
                               encoding="utf-8")
print(g.to_markdown())
