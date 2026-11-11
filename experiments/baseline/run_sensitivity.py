# E8: parametre hassasiyeti (makale Sekil 4)
#
# Her seferinde tek parametre degisiyor, digerleri varsayilan (c secimi Com/Sep,
# m = 2.3, b = 0.15, r_self = 0.05). 5 tekrar.
#   c (FCM kume sayisi), m (bulaniklik katsayisi) -> BoT-IoT ve Skin
#   b (alt populasyon esigi, Denk. 20), r_self (self yaricapi) -> BoT-IoT
# Makale Sekil 4'u NSL-KDD'de yapmis, bizde NSL-KDD yok (ASSUMPTIONS A14).
#
# calistirma: python -m experiments.baseline.run_sensitivity
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_botiot, load_skin, split_train_test
from src.metrics import evaluate
from src.naim import NaIM, NaIMConfig

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
N_RUNS = 5

GRID = {
    "fcm_c": [2, 3, 4, 5, 6, 8, 10],
    "fcm_m": [1.5, 1.8, 2.0, 2.3, 2.6, 3.0],
    "b_threshold": [0.05, 0.1, 0.15, 0.2, 0.3],
    "r_self": [0.01, 0.02, 0.05, 0.08, 0.12],
}
LABEL = {"fcm_c": "FCM kume sayisi c", "fcm_m": "bulaniklik katsayisi m",
         "b_threshold": "alt populasyon esigi b", "r_self": "self yaricapi r_self"}


def sweep(ds, frac, params, seed0, max_test=None):
    rows = []
    for param in params:
        for val in GRID[param]:
            for run_id in range(N_RUNS):
                rng = np.random.default_rng(seed0 + run_id)      # ayni tohum -> ayni bolunme
                S, N, X_te, y_te = split_train_test(ds, frac, rng)
                if max_test and len(y_te) > max_test:
                    keep = rng.permutation(len(y_te))[:max_test]
                    X_te, y_te = X_te[keep], y_te[keep]
                kw = {param: val}
                if param == "fcm_m":
                    kw["fcm_c"] = 5          # m taranirken c sabit
                t = time.perf_counter()
                m = NaIM(NaIMConfig(**kw)).fit(S, N, rng)
                dtt = time.perf_counter() - t
                rows.append({"data": ds.name, "param": param, "value": val, "run": run_id,
                             **evaluate(y_te, m.predict(X_te)), "Dn": m.info.n_detectors,
                             "DTT": dtt, "n_dist": m.info.n_dist})
        print("tamam:", ds.name, param)
    return rows


def plot(df):
    bot = df[df["data"] == "bot-iot-clean"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, p in zip(axes, GRID):
        g = bot[bot["param"] == p].groupby("value")["F1"].agg(["mean", "std"]).reset_index()
        ax.fill_between(g["value"], g["mean"] - g["std"], g["mean"] + g["std"], color="#2a78d6", alpha=0.15, lw=0)
        ax.plot(g["value"], g["mean"], color="#2a78d6", lw=2, marker="o", ms=5)
        ax.set_xlabel(LABEL[p])
        ax.grid(True, color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("F1 % (BoT-IoT, ort. ± std)")
    fig.tight_layout()
    fig.savefig(FIG / "e8_sensitivity.png", dpi=200)
    plt.close(fig)

    skin = df[(df["data"] == "skin") & (df["param"] == "fcm_c")]
    g = skin.groupby("value")["n_dist"].mean() / 1e6
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(g.index, g.values, color="#eb6834", lw=2, marker="o", ms=5)
    ax.set_xlabel("FCM kume sayisi c")
    ax.set_ylabel("uzaklik hesabi (milyon, Skin)")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "e8_cost_vs_c.png", dpi=200)
    plt.close(fig)


def table(df):
    out = []
    for (d, p), sub in df.groupby(["data", "param"], sort=False):
        g = sub.groupby("value")[["DR", "FAR", "F1", "Dn", "n_dist", "DTT"]].mean()
        g["n_dist"] = (g["n_dist"] / 1e6).round(2)
        out += [f"### {d} - {LABEL[p]}", "", g.round(2).rename(columns={"n_dist": "n_dist (M)"}).to_markdown(), ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sweep(load_botiot("clean"), 0.5, list(GRID), 800)
    rows += sweep(load_skin(), 0.02, ["fcm_c", "fcm_m"], 900, max_test=50000)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e8_sensitivity_runs.csv", index=False)
    (OUT / "e8_sensitivity.md").write_text("## E8 parametre hassasiyeti (5 tekrar)\n\n" + table(df), encoding="utf-8")
    plot(df)
    print(table(df))
