"""E1 - Iris yeniden üretim deneyi (makale Bölüm 4.3.1, Tablo 7).

Her Iris sınıfı sırayla self, diğer ikisi non-self kabul edilir. Self eğitim oranı
%50 ve %100; her ayar 20 bağımsız tekrar. Test, veri setinin tamamı üzerinde yapılır.

Karşılaştırılanlar:
  V-Detector  -- baseline (yalnızca self örnekler)
  NaIM-IDS    -- bizim yeniden gerçekleştirmemiz (self + aynı oranda non-self)

Çıktılar:
  results/tables/e1_iris_runs.csv     her tekrarın ham sonuçları
  results/tables/e1_iris_summary.csv  ortalama ve standart sapma
  results/tables/e1_iris_vs_paper.md  makale Tablo 7 ile yan yana karşılaştırma

Çalıştırma:  python -m experiments.baseline.run_iris
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_iris_self, split_self_train
from src.metrics import evaluate
from src.naim import NaIM, NaIMConfig
from src.vdetector import VDetectorConfig, train_vdetector

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_RUNS = 20
CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
FRACS = (0.5, 1.0)

# Makale Tablo 7 (ortalamalar): DR %, FAR %, Dn, DTT s
PAPER = {
    ("Setosa", 0.5):     {"V-Detector": (100.0, 3.2, 3653.1, 1.230), "NaIM-IDS": (100.0, 3.6, 151.0, 0.230)},
    ("Setosa", 1.0):     {"V-Detector": (100.0, 0.0, 3273.9, 1.020), "NaIM-IDS": (100.0, 0.0, 168.0, 0.033)},
    ("Versicolor", 0.5): {"V-Detector": (96.5, 8.6, 3568.4, 1.660),  "NaIM-IDS": (99.5, 3.3, 260.3, 0.045)},
    ("Versicolor", 1.0): {"V-Detector": (93.1, 0.0, 3568.0, 1.290),  "NaIM-IDS": (100.0, 0.0, 332.0, 0.047)},
    ("Virginica", 0.5):  {"V-Detector": (96.4, 10.0, 2822.3, 0.720), "NaIM-IDS": (99.6, 7.8, 408.0, 0.054)},
    ("Virginica", 1.0):  {"V-Detector": (95.1, 0.0, 2356.5, 0.500),  "NaIM-IDS": (100.0, 0.0, 497.0, 0.056)},
}


def run() -> pd.DataFrame:
    rows = []
    for cls, cname in CLASSES.items():
        ds = load_iris_self(cls)
        for frac in FRACS:
            for run_id in range(N_RUNS):
                rng = np.random.default_rng(1000 * cls + 10 * int(frac * 10) + run_id)
                X_self, X_non = split_self_train(ds, frac, rng)

                D, info = train_vdetector(X_self, VDetectorConfig(), rng)
                t = time.perf_counter(); pred = D.predict(ds.X); t_test = time.perf_counter() - t
                rows.append({"self": cname, "train_frac": frac, "run": run_id, "algo": "V-Detector",
                             **evaluate(ds.y, pred), "Dn": info.n_detectors,
                             "DTT": info.train_time, "Test": t_test, "n_dist": info.n_dist})

                model = NaIM(NaIMConfig()).fit(X_self, X_non, rng)
                t = time.perf_counter(); pred = model.predict(ds.X); t_test = time.perf_counter() - t
                rows.append({"self": cname, "train_frac": frac, "run": run_id, "algo": "NaIM-IDS",
                             **evaluate(ds.y, pred), "Dn": model.info.n_detectors,
                             "DTT": model.info.train_time, "Test": t_test,
                             "n_dist": model.info.n_dist, "fcm_c": model.info.extra["fcm_c"]})
            print(f"tamam: {cname} %{int(frac * 100)}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["DR", "FAR", "Dn", "DTT", "Test"]
    g = df.groupby(["self", "train_frac", "algo"], sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def paper_table(summary: pd.DataFrame) -> str:
    lines = ["| Self (eğitim) | Algoritma | DR % (bizim / makale) | FAR % (bizim / makale) "
             "| Dn (bizim / makale) | DTT s (bizim / makale) |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        p = PAPER[(r["self"], r["train_frac"])][r["algo"]]
        lines.append(
            f"| {r['self']} (%{int(r['train_frac'] * 100)}) | {r['algo']} "
            f"| {r['DR_mean']:.1f} ± {r['DR_std']:.1f} / {p[0]:.1f} "
            f"| {r['FAR_mean']:.1f} ± {r['FAR_std']:.1f} / {p[1]:.1f} "
            f"| {r['Dn_mean']:.1f} ± {r['Dn_std']:.1f} / {p[2]:.1f} "
            f"| {r['DTT_mean']:.3f} / {p[3]:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(OUT / "e1_iris_runs.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "e1_iris_summary.csv", index=False)
    (OUT / "e1_iris_vs_paper.md").write_text(paper_table(s), encoding="utf-8")
    print(paper_table(s))
