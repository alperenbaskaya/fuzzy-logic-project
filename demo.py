# Canli demo - sunumda kullanmak icin.
#
# 1) Elle girdi ver, sistemin nasil karar verdigini adim adim gor:
#      python demo.py --g1 1.1 --g2 0.4
#      python demo.py --g1 1.1 --g2 0.4 --mf gauss --inference sugeno
#      python demo.py --g1 1.1 --g2 0.4 --plot        (cikti kumesini resim olarak kaydeder)
#
# 2) Gercek veriden ornek: NaIM'i egitir, test orneklerinde keskin ve bulanik
#    karari yan yana gosterir:
#      python demo.py --data botiot --n 5
#      python demo.py --data skin --n 5 --seed 3
#      python demo.py --data iris --n 5
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.fuzzy_detect import INPUT1, INPUT2, OUTPUT, RULES, FuzzyConfig, FuzzyDetector, defuzzify

LINE = "-" * 64


def explain(fd, g1, g2, show_plot=False):
    cfg = fd.cfg
    r = fd.infer(g1, g2)
    print(LINE)
    print(f"Girdiler:  g1 (kapsama) = {g1:.3f}    g2 (self uzakligi) = {g2:.3f}")
    print(f"Ayarlar:   MF = {cfg.mf}, cikarim = {cfg.inference}, durulastirma = {cfg.defuzz}, esik = {cfg.threshold}")

    print("\n1) Bulaniklastirma (uyelik dereceleri)")
    print("   g1: " + "   ".join(f"{k} {r['mu1'][k][0]:.2f}" for k in INPUT1))
    print("   g2: " + "   ".join(f"{k} {r['mu2'][k][0]:.2f}" for k in INPUT2))

    print("\n2) Ateslenen kurallar (VE = min)")
    fired = [(j, r["w"][0, j]) for j in range(len(RULES)) if r["w"][0, j] >= 0.005]
    if not fired:
        print("   hic kural ateslenmedi -> skor 0.5 (kararsiz)")
    for j, w in sorted(fired, key=lambda t: -t[1]):
        a, b, c = RULES[j]
        print(f"   K{j + 1}: EGER g1 {a:<8} ({r['mu1'][a][0]:.2f}) VE g2 {b:<5} ({r['mu2'][b][0]:.2f})"
              f"  ISE {c:<7}  w = {w:.2f}")

    if cfg.inference == "mamdani":
        y, agg = r["y"], r["agg"]
        print("\n3) Birlestirilmis cikti kumesi (her kural kendi kumesini w seviyesinde kirpar, max ile birlesir)")
        cols = np.linspace(0, len(y) - 1, 41).astype(int)
        for level in (0.8, 0.6, 0.4, 0.2):
            print(f"   {level:.1f} |" + "".join("#" if agg[0, i] >= level else " " for i in cols))
        print("       +" + "-" * 41)
        print("        0        Normal      0.5    Supheli         Saldiri   1")
        print("\n4) Durulastirma")
        for m in ("centroid", "bisector", "mom"):
            v = defuzzify(y, agg, m)[0]
            mark = "  <- kullanilan" if m == cfg.defuzz else ""
            print(f"   {m:<9} = {v:.3f}{mark}")
    else:
        print("\n3) Sugeno: skor = sum(w * z) / sum(w),  z: Normal 0, Supheli 0.5, Saldiri 1")

    s = r["score"][0]
    label = "SALDIRI" if s >= cfg.threshold else "NORMAL"
    print(f"\nSonuc: anomali derecesi = {s:.3f}  ->  {label}")
    print(f"Keskin sistem (makale): g1 {'>=' if g1 >= 1 else '<'} 1  ->  {'SALDIRI' if g1 >= 1 else 'NORMAL'}")
    print(LINE)

    if show_plot and cfg.inference == "mamdani":
        plot_inference(fd, r, s)


def plot_inference(fd, r, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.fuzzy_detect import membership

    y = r["y"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    show = {"Normal": "Normal", "Supheli": "Şüpheli", "Saldiri": "Saldırı"}
    for color, (name, p) in zip(["#2a78d6", "#eb6834", "#1baf7a"], OUTPUT.items()):
        ax.plot(y, membership(y, p, fd.cfg.mf), color=color, lw=1.2, ls="--", label=show[name])
    ax.fill_between(y, r["agg"][0], color="#6b6a64", alpha=0.35, label="birleştirilmiş çıktı")
    ax.axvline(s, color="#1f1f1e", lw=2)
    ax.text(s, 1.02, f" {fd.cfg.defuzz} = {s:.2f}", fontsize=9)
    ax.set_xlabel("anomali derecesi")
    ax.set_ylabel("üyelik")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="center right", frameon=False)
    fig.tight_layout()
    out = Path(__file__).parent / "results" / "figures" / "demo_inference.png"
    fig.savefig(out, dpi=150)
    print(f"grafik kaydedildi: {out}")


def data_demo(args, cfg):
    from src.data import load_botiot, load_iris_self, load_skin, split_train_test
    from src.naim import NaIM, NaIMConfig

    if args.data == "botiot":
        ds, frac = load_botiot("clean"), 0.5
    elif args.data == "skin":
        ds, frac = load_skin(), 0.02
    else:
        ds, frac = load_iris_self(1), 0.5
    rng = np.random.default_rng(args.seed)
    S, N, X_te, y_te = split_train_test(ds, frac, rng)
    print(f"{ds.name}: egitim {len(S)} normal + {len(N)} saldiri, NaIM egitiliyor...")
    model = NaIM(NaIMConfig()).fit(S, N, rng)
    fd = FuzzyDetector.from_model(model, N, cfg)
    print(f"{model.info.n_detectors} dedektor uretildi, d_ref = {fd.d_ref:.3f}\n")

    # ilginc ornekler: once keskin ve bulanigin ayristigi ornekler, sonra rastgele
    g1, g2 = fd.inputs(X_te)
    s = fd.score_from_inputs(g1, g2)
    crisp, fuzzy = g1 >= 1, s >= cfg.threshold
    diff = np.flatnonzero(crisp != fuzzy)
    pick = list(rng.permutation(diff)[: args.n // 2 + 1]) if len(diff) else []
    rest = [i for i in rng.permutation(len(y_te)) if i not in pick]
    pick = (pick + rest)[: args.n]

    for i in pick:
        print(f"\nTest ornegi #{i}   gercek etiket: {'SALDIRI' if y_te[i] else 'NORMAL'}")
        explain(fd, float(g1[i]), float(g2[i]))
    print(f"\nTum testte keskin ile bulanigin farkli karar verdigi ornek sayisi: {len(diff)} / {len(y_te)}")


def main():
    ap = argparse.ArgumentParser(description="Bulanik tespit katmani canli demo")
    ap.add_argument("--g1", type=float, help="kapsama orani, 0-2 (1 = dedektor siniri)")
    ap.add_argument("--g2", type=float, help="self uzakligi / d_ref, 0-2")
    ap.add_argument("--data", choices=["botiot", "skin", "iris"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mf", choices=["tri", "gauss"], default="tri")
    ap.add_argument("--inference", choices=["mamdani", "sugeno"], default="mamdani")
    ap.add_argument("--defuzz", choices=["centroid", "bisector", "mom"], default="centroid")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = FuzzyConfig(args.mf, args.inference, args.defuzz, args.threshold)

    if args.data:
        data_demo(args, cfg)
    elif args.g1 is not None and args.g2 is not None:
        explain(FuzzyDetector(None, None, 1.0, cfg), args.g1, args.g2, args.plot)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
