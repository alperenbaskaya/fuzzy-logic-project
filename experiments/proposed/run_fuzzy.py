# E4/E5/E7 bulanik katman deneyleri - BoT-IoT ve Skin icin (Iris icin run_fuzzy_iris.py)
#
# Her tekrarda: egitim kismi 70/30 bolunur. NaIM 70'te egitilir, 30 validasyon.
#   - keskin karar (makale)
#   - kapsama skoru (FIS yok) + validasyonda secilen esik
#   - FIS varyantlari: esik 0.5 ve validasyonda secilen esik
#   - gurultu testi (E7)
# Skin test kumesi cok buyuk (240 bin), FIS'te her ornek icin en yakin self'i
# aradigim icin testi 20 bin ornege indiriyorum.
#
# calistirma: python -m experiments.proposed.run_fuzzy
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from experiments.proposed.run_fuzzy_iris import NOISE, VARIANTS, best_threshold
from src.data import load_botiot, load_skin, split_train_test
from src.fuzzy_detect import FuzzyDetector
from src.metrics import evaluate
from src.naim import NaIM, NaIMConfig

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_RUNS = 10
MAIN = "FIS ucgen mamdani centroid"


def auc(y, s):
    return 100 * roc_auc_score(y, s)


def run(ds, frac, seed0, max_test=None):
    rows, noise_rows = [], []
    for run_id in range(N_RUNS):
        rng = np.random.default_rng(seed0 + run_id)
        S, N, X_te, y_te = split_train_test(ds, frac, rng)
        if max_test and len(y_te) > max_test:
            keep = rng.permutation(len(y_te))[:max_test]
            X_te, y_te = X_te[keep], y_te[keep]

        ps, pn = rng.permutation(len(S)), rng.permutation(len(N))
        ks, kn = int(0.7 * len(S)), int(0.7 * len(N))
        S_tr, N_tr = S[ps[:ks]], N[pn[:kn]]
        X_va = np.vstack([S[ps[ks:]], N[pn[kn:]]])
        y_va = np.r_[np.zeros(len(S) - ks), np.ones(len(N) - kn)]

        model = NaIM(NaIMConfig()).fit(S_tr, N_tr, rng)
        fd = FuzzyDetector.from_model(model, N_tr)
        g1_va, g2_va = fd.inputs(X_va)
        g1_te, g2_te = fd.inputs(X_te)
        base = {"data": ds.name, "run": run_id}

        crisp = model.predict(X_te)
        rows.append({**base, "method": "Keskin (makale)", **evaluate(y_te, crisp), "AUC": auc(y_te, crisp)})
        t = best_threshold(y_va, g1_va)
        rows.append({**base, "method": "Kapsama skoru + ayarli esik",
                     **evaluate(y_te, g1_te >= t), "AUC": auc(y_te, g1_te)})

        for name, cfg in VARIANTS.items():
            fd.cfg = cfg
            s_te = fd.score_from_inputs(g1_te, g2_te)
            rows.append({**base, "method": f"{name} (esik 0.5)", **evaluate(y_te, s_te >= 0.5),
                         "AUC": auc(y_te, s_te)})
            t = best_threshold(y_va, fd.score_from_inputs(g1_va, g2_va))
            rows.append({**base, "method": f"{name} + ayarli esik", "esik": t,
                         **evaluate(y_te, s_te >= t), "AUC": auc(y_te, s_te)})

        # E7 gurultu: ana FIS, validasyonda secilen esikle
        fd.cfg = VARIANTS[MAIN]
        t_main = best_threshold(y_va, fd.score_from_inputs(g1_va, g2_va))
        for sigma in NOISE:
            nrng = np.random.default_rng(9000 + run_id)
            Xn = np.clip(X_te + nrng.normal(0, sigma, X_te.shape), 0, 1)
            noise_rows.append({**base, "sigma": sigma, "method": "Keskin", **evaluate(y_te, model.predict(Xn))})
            noise_rows.append({**base, "sigma": sigma, "method": "Bulanik",
                               **evaluate(y_te, fd.score(Xn) >= t_main)})
    print("tamam:", ds.name)
    return rows, noise_rows


def table(df):
    out = []
    for d, sub in df.groupby("data", sort=False):
        g = sub.groupby("method", sort=False)[["DR", "FAR", "F1", "AUC"]].agg(["mean", "std"])
        out += [f"### {d}", "", "| Yontem | DR % | FAR % | F1 % | AUC % |", "|---|---|---|---|---|"]
        for m, r in g.iterrows():
            out.append(f"| {m} | " + " | ".join(
                f"{r[(k, 'mean')]:.2f} ± {r[(k, 'std')]:.2f}" for k in ("DR", "FAR", "F1", "AUC")) + " |")
        a = sub[sub["method"] == "Keskin (makale)"].sort_values("run")
        out += ["", "| Karsilastirma (vs keskin) | Metrik | Ort. fark | p |", "|---|---|---|---|"]
        for m in (f"{MAIN} (esik 0.5)", f"{MAIN} + ayarli esik", "FIS gauss sugeno + ayarli esik",
                  "Kapsama skoru + ayarli esik"):
            b = sub[sub["method"] == m].sort_values("run")
            for k in ("DR", "FAR", "F1"):
                diff = b[k].to_numpy() - a[k].to_numpy()
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                out.append(f"| {m} | {k} | {diff.mean():+.2f} | {p:.4f} |")
        # FIS'in kendi katkisi: ayni sekilde esigi ayarlanmis FIS'siz skora gore
        c = sub[sub["method"] == "Kapsama skoru + ayarli esik"].sort_values("run")
        out += ["", "| Karsilastirma (vs kapsama skoru + ayarli esik) | Metrik | Ort. fark | p |",
                "|---|---|---|---|"]
        for m in (f"{MAIN} + ayarli esik", "FIS gauss sugeno + ayarli esik"):
            b = sub[sub["method"] == m].sort_values("run")
            for k in ("F1", "FAR", "AUC"):
                diff = b[k].to_numpy() - c[k].to_numpy()
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                out.append(f"| {m} | {k} | {diff.mean():+.2f} | {p:.4f} |")
        out.append("")
    return "\n".join(out) + "\n"


def noise_table(nf):
    out = []
    for d, sub in nf.groupby("data", sort=False):
        g = sub.groupby(["sigma", "method"])[["DR", "FAR", "F1"]].mean().unstack("method").round(2)
        out += [f"### {d}", "", g.to_markdown(), ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    r1, n1 = run(load_botiot("clean"), 0.5, 400)
    r2, n2 = run(load_skin(), 0.02, 600, max_test=20000)
    df, nf = pd.DataFrame(r1 + r2), pd.DataFrame(n1 + n2)
    df.to_csv(OUT / "e4_fuzzy_runs.csv", index=False)
    nf.to_csv(OUT / "e7_noise_runs.csv", index=False)
    text = ("## E4/E5 bulanik katman (10 tekrar)\n\n" + table(df)
            + "\n## E7 gurultu (validasyonda secilen esik)\n\n" + noise_table(nf))
    (OUT / "e4_e7_fuzzy.md").write_text(text, encoding="utf-8")
    print(text)
