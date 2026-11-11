# Rapor/sunum grafikleri:
#   fis_membership.png  - giris/cikis uyelik fonksiyonlari (ucgen ve gauss)
#   fis_surface.png     - kontrol yuzeyi (mamdani centroid)
#   e7_noise.png        - gurultu altinda F1 / FAR (keskin vs bulanik)
# calistirma: python -m experiments.proposed.plot_figures  (once run_fuzzy_iris)
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fuzzy_detect import INPUT1, INPUT2, OUTPUT, FuzzyConfig, FuzzyDetector, membership

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # mavi, turuncu, yesil
# grafiklerde Turkce karakterli gosterim adlari
SHOW = {"Disarida": "Dışarıda", "Sinirda": "Sınırda", "Icerde": "İçeride", "Yakin": "Yakın",
        "Orta": "Orta", "Uzak": "Uzak", "Normal": "Normal", "Supheli": "Şüpheli", "Saldiri": "Saldırı"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
})


def plot_membership():
    sets = [(INPUT1, "Girdi 1: kapsama oranı g1", (0, 2)),
            (INPUT2, "Girdi 2: self uzaklığı g2", (0, 2)),
            (OUTPUT, "Çıktı: anomali derecesi", (0, 1))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharey=True)
    for row, mf in enumerate(("tri", "gauss")):
        for col, (terms, title, (lo, hi)) in enumerate(sets):
            ax = axes[row, col]
            x = np.linspace(lo, hi, 400)
            for color, (name, params) in zip(SERIES, terms.items()):
                ax.plot(x, membership(x, params, mf), color=color, lw=2, label=SHOW[name])
            ax.set_ylim(-0.03, 1.05)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=10)
            if col == 0:
                ax.set_ylabel(("Üçgen" if mf == "tri" else "Gauss") + "\nüyelik derecesi")
            ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fis_membership.png")
    plt.close(fig)


def plot_surface():
    fd = FuzzyDetector(None, None, 1.0, FuzzyConfig("tri", "mamdani", "centroid"))
    g = np.linspace(0, 2, 121)
    G1, G2 = np.meshgrid(g, g)
    S = fd.score_from_inputs(G1.ravel(), G2.ravel()).reshape(G1.shape)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(S, origin="lower", extent=(0, 2, 0, 2), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    # skor bircok yerde tam 0.5'te sabit (Supheli platosu), o yuzden karar bolgesinin
    # kendisini ciziyorum: skor >= 0.5 -> saldiri
    dec = (S >= 0.5 - 1e-9).astype(float)
    ax.contour(G1, G2, dec, levels=[0.5], colors=INK, linewidths=1.4)
    ax.text(0.15, 0.15, "NORMAL", color=INK, fontsize=9, weight="bold")
    ax.text(1.55, 1.75, "SALDIRI", color="white", fontsize=9, weight="bold")
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.text(1.02, 0.6, "keskin karar\nsınırı (g1 = 1)", color=MUTED, fontsize=8)
    ax.set_xlabel("g1: kapsama oranı (1 ve üstü = kürenin içinde)")
    ax.set_ylabel("g2: self uzaklığı / d_ref")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="anomali derecesi")
    fig.tight_layout()
    fig.savefig(FIG / "fis_surface.png")
    plt.close(fig)


def plot_noise():
    nf = pd.read_csv(TAB / "e7_noise_iris_runs.csv")
    g = nf.groupby(["sigma", "method"])[["F1", "FAR"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, metric, label in ((axes[0], "F1", "F1 (%)"), (axes[1], "FAR", "Yanlış alarm oranı (%)")):
        for color, m, name in ((SERIES[0], "Keskin", "Keskin (makale)"), (SERIES[1], "Bulanik", "Bulanık (FIS)")):
            d = g[g["method"] == m]
            ax.plot(d["sigma"], d[metric], color=color, lw=2, marker="o", ms=5, label=name)
            ax.annotate(name, (d["sigma"].iloc[-1], d[metric].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", color=INK, fontsize=8, va="center")
        ax.set_xlabel("test verisine eklenen gürültü (σ)")
        ax.set_ylabel(label)
        ax.set_xlim(-0.005, 0.16)
    axes[1].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=9)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "e7_noise.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    plot_membership()
    plot_surface()
    plot_noise()
    print("grafikler:", sorted(p.name for p in FIG.glob("*.png")))
