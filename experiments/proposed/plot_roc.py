# ROC egrileri: keskin karar tek bir nokta, surekli skorlar egri verir.
# Skin ve BoT-IoT icin run_fuzzy.py'deki ilk tekrarla ayni bolunme (tohum 400 / 600).
# calistirma: python -m experiments.proposed.plot_roc
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import load_botiot, load_skin, split_train_test
from src.fuzzy_detect import FuzzyConfig, FuzzyDetector
from src.naim import NaIM, NaIMConfig

FIG = Path(__file__).resolve().parents[2] / "results" / "figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLORS = {"Kapsama skoru (FIS yok)": "#6b6a64", "FIS ucgen Mamdani": "#2a78d6", "FIS Gauss Sugeno": "#eb6834"}


def one(ds, frac, seed, max_test=None):
    # run_fuzzy.py ile ayni adimlar (70/30 bolme dahil) ki egriler tablodaki ilk tekrarla ayni olsun
    rng = np.random.default_rng(seed)
    S, N, X_te, y_te = split_train_test(ds, frac, rng)
    if max_test and len(y_te) > max_test:
        keep = rng.permutation(len(y_te))[:max_test]
        X_te, y_te = X_te[keep], y_te[keep]
    ps, pn = rng.permutation(len(S)), rng.permutation(len(N))
    S_tr, N_tr = S[ps[:int(0.7 * len(S))]], N[pn[:int(0.7 * len(N))]]
    model = NaIM(NaIMConfig()).fit(S_tr, N_tr, rng)
    fd = FuzzyDetector.from_model(model, N_tr)
    g1, g2 = fd.inputs(X_te)
    scores = {"Kapsama skoru (FIS yok)": g1}
    fd.cfg = FuzzyConfig("tri", "mamdani")
    scores["FIS ucgen Mamdani"] = fd.score_from_inputs(g1, g2)
    fd.cfg = FuzzyConfig("gauss", "sugeno")
    scores["FIS Gauss Sugeno"] = fd.score_from_inputs(g1, g2)
    crisp = model.predict(X_te)
    return y_te, scores, crisp


fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
# her panelin ekseni kendi verisine gore (egriler sol ust kosede)
for ax, (title, ds, frac, seed, mt, xl, yl) in zip(axes, [
        ("Skin", load_skin(), 0.02, 600, 20000, 1.5, 97.0),
        ("BoT-IoT", load_botiot("clean"), 0.5, 400, None, 12.0, 93.0)]):
    y, scores, crisp = one(ds, frac, seed, mt)
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(100 * fpr, 100 * tpr, color=COLORS[name], lw=2,
                label=f"{name} (AUC {100 * roc_auc_score(y, s):.2f})")
    fp = np.mean(crisp[y == 0]); tp = np.mean(crisp[y == 1])
    ax.plot(100 * fp, 100 * tp, "o", ms=9, color=INK, mec="white", mew=2, zorder=5,
            label=f"Keskin karar (tek nokta)")
    ax.set_xlim(-0.02 * xl, xl)
    ax.set_ylim(yl, 100.2)
    ax.set_title(f"{title} (ilk tekrar, eksenler yakinlastirildi)", color=INK, fontsize=10)
    ax.set_xlabel("yanlis alarm orani FAR (%)")
    ax.set_ylabel("tespit orani DR (%)")
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "roc_curves.png", dpi=200)
print("kaydedildi:", FIG / "roc_curves.png")
