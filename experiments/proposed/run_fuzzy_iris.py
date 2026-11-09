# E4 + E5 + E7: keskin ve bulanik tespitin karsilastirmasi (Iris)
#
# Her Iris sinifi sirayla self, %50 egitim, 20 tekrar.
# Test: egitimde KULLANILMAYAN ornekler (E1'den farkli olarak, daha adil).
# Ayni egitilmis NaIM dedektorleri uzerinde:
#   Keskin            -> kurenin icinde mi (makale)
#   Kapsama skoru     -> sadece g1, FIS yok (FIS'in katkisini ayirmak icin, sadece AUC)
#   FIS varyantlari   -> MF tipi, Mamdani/Sugeno, durulastirma yontemi
# E7: test verisine Gauss gurultusu ekleyip keskin vs bulanik
#
# calistirma: python -m experiments.proposed.run_fuzzy_iris
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from src.data import load_iris_self, split_train_test
from src.fuzzy_detect import FuzzyConfig, FuzzyDetector
from src.metrics import evaluate
from src.naim import NaIM, NaIMConfig

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_RUNS = 20
FRAC = 0.5
NOISE = [0.0, 0.02, 0.05, 0.08, 0.12]

VARIANTS = {
    "FIS ucgen mamdani centroid": FuzzyConfig("tri", "mamdani", "centroid"),
    "FIS gauss mamdani centroid": FuzzyConfig("gauss", "mamdani", "centroid"),
    "FIS ucgen mamdani bisector": FuzzyConfig("tri", "mamdani", "bisector"),
    "FIS ucgen mamdani mom":      FuzzyConfig("tri", "mamdani", "mom"),
    "FIS ucgen sugeno":           FuzzyConfig("tri", "sugeno"),
    "FIS gauss sugeno":           FuzzyConfig("gauss", "sugeno"),
}
MAIN = "FIS ucgen mamdani centroid"


def safe_auc(y, s):
    return 100 * roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def run():
    rows, noise_rows = [], []
    for cls in range(3):
        ds = load_iris_self(cls)
        for run_id in range(N_RUNS):
            rng = np.random.default_rng(5000 + 100 * cls + run_id)
            X_self, X_non, X_te, y_te = split_train_test(ds, FRAC, rng)
            model = NaIM(NaIMConfig()).fit(X_self, X_non, rng)
            base = {"self": ds.name, "run": run_id}

            # keskin karar
            crisp = model.predict(X_te)
            rows.append({**base, "method": "Keskin (makale)", **evaluate(y_te, crisp),
                         "AUC": safe_auc(y_te, crisp)})

            # FIS'siz surekli skor (sadece AUC anlamli)
            fd = FuzzyDetector.from_model(model, X_non)
            g1, g2 = fd.inputs(X_te)
            rows.append({**base, "method": "Kapsama skoru (FIS yok)",
                         **evaluate(y_te, g1 >= 1), "AUC": safe_auc(y_te, g1)})

            # FIS varyantlari (girdiler bir kere hesaplaniyor)
            for name, cfg in VARIANTS.items():
                fd.cfg = cfg
                s = fd.score_from_inputs(g1, g2)
                rows.append({**base, "method": name, **evaluate(y_te, s >= cfg.threshold),
                             "AUC": safe_auc(y_te, s)})

            # E7: gurultu
            fd.cfg = VARIANTS[MAIN]
            for sigma in NOISE:
                nrng = np.random.default_rng(9000 + run_id)
                Xn = np.clip(X_te + nrng.normal(0, sigma, X_te.shape), 0, 1)
                noise_rows.append({**base, "sigma": sigma, "method": "Keskin",
                                   **evaluate(y_te, model.predict(Xn))})
                noise_rows.append({**base, "sigma": sigma, "method": "Bulanik",
                                   **evaluate(y_te, fd.predict(Xn))})
        print("tamam:", ds.name)
    return pd.DataFrame(rows), pd.DataFrame(noise_rows)


def best_threshold(y, s):
    # validasyon setinde F1'i en buyuk yapan esik
    best_t, best_f = 0.5, -1
    for t in np.unique(s):
        f = evaluate(y, s >= t)["F1"]
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def run_tuned():
    # E4b: esigi test verisine bakmadan ayarla.
    # egitim kismini 70/30 bol: 70 ile NaIM egit, 30'da esik sec, ayrik testte olc.
    # keskin karar da ayni modeli kullaniyor, yani karsilastirma adil.
    rows = []
    for cls in range(3):
        ds = load_iris_self(cls)
        for run_id in range(N_RUNS):
            rng = np.random.default_rng(7000 + 100 * cls + run_id)
            X_self, X_non, X_te, y_te = split_train_test(ds, FRAC, rng)
            ks, kn = int(0.7 * len(X_self)), int(0.7 * len(X_non))
            ps, pn = rng.permutation(len(X_self)), rng.permutation(len(X_non))
            S_tr, S_va = X_self[ps[:ks]], X_self[ps[ks:]]
            N_tr, N_va = X_non[pn[:kn]], X_non[pn[kn:]]
            X_va = np.vstack([S_va, N_va])
            y_va = np.r_[np.zeros(len(S_va)), np.ones(len(N_va))]

            model = NaIM(NaIMConfig()).fit(S_tr, N_tr, rng)
            fd = FuzzyDetector.from_model(model, N_tr, VARIANTS[MAIN])
            base = {"self": ds.name, "run": run_id}

            rows.append({**base, "method": "Keskin (makale)", **evaluate(y_te, model.predict(X_te))})

            g1_va, _ = fd.inputs(X_va)
            g1_te, _ = fd.inputs(X_te)
            t = best_threshold(y_va, g1_va)
            rows.append({**base, "method": "Kapsama skoru + ayarli esik", **evaluate(y_te, g1_te >= t)})

            for name in (MAIN, "FIS ucgen sugeno"):
                fd.cfg = VARIANTS[name]
                t = best_threshold(y_va, fd.score(X_va))
                rows.append({**base, "method": f"{name} + ayarli esik", "esik": t,
                             **evaluate(y_te, fd.score(X_te) >= t)})
        print("tamam (esik ayari):", ds.name)
    return pd.DataFrame(rows)


def tuned_table(tf):
    g = tf.groupby("method", sort=False)[["DR", "FAR", "F1"]].agg(["mean", "std"])
    lines = ["| Yontem | DR % | FAR % | F1 % |", "|---|---|---|---|"]
    for m, r in g.iterrows():
        lines.append(f"| {m} | " + " | ".join(
            f"{r[(k, 'mean')]:.1f} ± {r[(k, 'std')]:.1f}" for k in ("DR", "FAR", "F1")) + " |")
    a = tf[tf["method"] == "Keskin (makale)"].sort_values(["self", "run"])
    lines += ["", "| Karsilastirma | Metrik | Ort. fark | p-degeri |", "|---|---|---|---|"]
    for m in tf["method"].unique()[1:]:
        b = tf[tf["method"] == m].sort_values(["self", "run"])
        for k in ("DR", "FAR", "F1"):
            d = b[k].to_numpy() - a[k].to_numpy()
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            lines.append(f"| {m} vs keskin | {k} | {d.mean():+.2f} | {p:.4f} |")
    return "\n".join(lines) + "\n"


def summary_table(df):
    order = ["Keskin (makale)", "Kapsama skoru (FIS yok)"] + list(VARIANTS)
    g = df.groupby("method")[["DR", "FAR", "F1", "AUC"]].agg(["mean", "std"]).reindex(order)
    lines = ["| Yontem | DR % | FAR % | F1 % | AUC % |", "|---|---|---|---|---|"]
    for m, r in g.iterrows():
        cells = [f"{r[(k, 'mean')]:.1f} ± {r[(k, 'std')]:.1f}" for k in ("DR", "FAR", "F1", "AUC")]
        if m == "Kapsama skoru (FIS yok)":
            cells[:3] = ["(= keskin)"] * 3
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def per_class_table(df):
    sub = df[df["method"].isin(["Keskin (makale)", MAIN])]
    g = sub.groupby(["self", "method"])[["DR", "FAR", "F1", "AUC"]].mean().round(1)
    return g.to_markdown() + "\n"


def wilcoxon_table(df):
    a = df[df["method"] == "Keskin (makale)"].sort_values(["self", "run"])
    lines = ["| Karsilastirma | Metrik | Ort. fark (bulanik - keskin) | p-degeri |", "|---|---|---|---|"]
    for name in [MAIN, "FIS ucgen sugeno"]:
        b = df[df["method"] == name].sort_values(["self", "run"])
        for k in ("DR", "FAR", "F1"):
            d = b[k].to_numpy() - a[k].to_numpy()
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            lines.append(f"| {name} vs keskin | {k} | {d.mean():+.2f} | {p:.4f} |")
    return "\n".join(lines) + "\n"


def noise_table(nf):
    g = nf.groupby(["sigma", "method"])[["DR", "FAR", "F1"]].mean().unstack("method").round(1)
    return g.to_markdown() + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df, nf = run()
    tf = run_tuned()
    df.to_csv(OUT / "e4_fuzzy_iris_runs.csv", index=False)
    nf.to_csv(OUT / "e7_noise_iris_runs.csv", index=False)
    tf.to_csv(OUT / "e4b_tuned_iris_runs.csv", index=False)
    text = ("## E4/E5 ozet (3 sinif x 20 tekrar, %50 egitim, ayrik test)\n\n" + summary_table(df)
            + "\n## Sinif bazinda\n\n" + per_class_table(df)
            + "\n## Wilcoxon isaretli sira testi (n = 60)\n\n" + wilcoxon_table(df)
            + "\n## E4b esik validasyonla ayarlandiginda (egitim 70/30)\n\n" + tuned_table(tf)
            + "\n## E7 gurultu (ortalama)\n\n" + noise_table(nf))
    (OUT / "e4_e7_fuzzy_iris.md").write_text(text, encoding="utf-8")
    print(text)
