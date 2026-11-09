# E2: Skin (makale Tablo 8), E3: BoT-IoT (makale Tablo 5), E6: kumeleme ablasyonu
#
# E2 Skin: deri = self. Makale egitim oranini yazmamis; her siniftan %2 ile
#          egitiyorum (ASSUMPTIONS A11), geri kalan ~240 bin ornekte test.
# E3 BoT-IoT: 477 normal + 5050 saldiri alt kumesi, %50 egitim / %50 test.
#          iki ozellik kumesi: makalenin Tablo 3 listesi ve sizintisiz "clean" liste
# E6: NaIM'i FCM / K-means / kumelemesiz calistir. budama tam oldugu icin
#          dogruluk degismemeli, sadece uzaklik hesabi sayisi (n_dist) ve sure degismeli
#
# calistirma: python -m experiments.baseline.run_skin_botiot
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_botiot, load_skin, split_train_test
from src.metrics import evaluate
from src.naim import NaIM, NaIMConfig
from src.vdetector import VDetectorConfig, train_vdetector

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_RUNS = 10

# makale Tablo 8 (Skin) ve Tablo 5 (BoT-IoT)
PAPER_SKIN = {"V-Detector": (87.20, 3.04, 375.9), "NaIM-IDS": (98.55, 3.01, 248.2)}
PAPER_BOTIOT = {"Accuracy": 98.60, "DR": 97.73, "Precision": 99.24, "F1": 98.47}


def one(name, fit, X_te, y_te):
    t = time.perf_counter()
    model, info = fit()
    dtt = time.perf_counter() - t
    t = time.perf_counter()
    pred = model.predict(X_te)
    return {"algo": name, **evaluate(y_te, pred), "Dn": info.n_detectors, "DTT": dtt,
            "Test": time.perf_counter() - t, "n_dist": info.n_dist}


def naim_fit(S, N, rng, **kw):
    def f():
        m = NaIM(NaIMConfig(**kw)).fit(S, N, rng)
        return m, m.info
    return f


def vd_fit(S, rng):
    return lambda: train_vdetector(S, VDetectorConfig(), rng)


def run_dataset(ds, frac, seed0, ablation=False):
    rows = []
    for run_id in range(N_RUNS):
        rng = np.random.default_rng(seed0 + run_id)
        S, N, X_te, y_te = split_train_test(ds, frac, rng)
        base = {"data": ds.name, "run": run_id}
        rows.append({**base, **one("V-Detector", vd_fit(S, rng), X_te, y_te)})
        rows.append({**base, **one("NaIM-IDS", naim_fit(S, N, rng), X_te, y_te)})
        if ablation:
            # E6: ayni tohumla, sadece kumeleme degisiyor (c = 5, makale Sekil 4a)
            for cl in ("fcm", "kmeans", "none"):
                r = np.random.default_rng(seed0 + 500 + run_id)
                rows.append({**base, **one(f"NaIM ({cl}, c=5)",
                                           naim_fit(S, N, r, cluster=cl, fcm_c=5), X_te, y_te)})
    print("tamam:", ds.name)
    return rows


def fmt(df, cols):
    g = df.groupby(["data", "algo"], sort=False)[cols].agg(["mean", "std"])
    lines = ["| Veri | Algoritma | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for (d, a), r in g.iterrows():
        cells = []
        for c in cols:
            m, s = r[(c, "mean")], r[(c, "std")]
            cells.append(f"{m:.0f} ± {s:.0f}" if c in ("Dn", "n_dist") else
                         f"{m:.3f}" if c in ("DTT", "Test") else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {d} | {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run_dataset(load_skin(), 0.02, 100, ablation=True)
    rows += run_dataset(load_botiot("paper"), 0.5, 200)
    rows += run_dataset(load_botiot("clean"), 0.5, 300, ablation=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2_e3_e6_runs.csv", index=False)

    main = df[df["algo"].isin(["V-Detector", "NaIM-IDS"])]
    abl = df[df["algo"].str.contains(r"c=5")]
    text = (
        "## E2 Skin ve E3 BoT-IoT (10 tekrar)\n\n"
        + fmt(main, ["DR", "FAR", "Accuracy", "Precision", "F1", "Dn", "DTT", "Test"])
        + "\nMakale Tablo 8 (Skin): V-Detector DR 87.20, FAR 3.04, Dn 375.9; "
          "NaIM-IDS DR 98.55, FAR 3.01, Dn 248.2\n"
        + "Makale Tablo 5 (BoT-IoT): Accuracy 98.60, Recall 97.73, Precision 99.24, F1 98.47\n"
        + "\n## E6 kumeleme ablasyonu (c = 5)\n\n"
        + fmt(abl, ["DR", "FAR", "F1", "Dn", "DTT", "n_dist"])
    )
    (OUT / "e2_e3_e6.md").write_text(text, encoding="utf-8")
    print(text)
