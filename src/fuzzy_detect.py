# Bulanik tespit katmani - projedeki kendi katkim.
#
# Makalede karar keskin: ornek bir kurenin icindeyse saldiri, degilse normal.
# Kurenin hemen disindaki bir saldiri da, kureden cok uzaktaki bir normal ornek de
# ayni sekilde "0" aliyor. Ben bunun yerine iki girdiden bir "anomali derecesi" (0-1)
# cikaran bir bulanik cikarim sistemi (FIS) yazdim. Kutuphane kullanmadim.
#
# Girdi 1 - kapsama (g1): max_i r_i / ||x - c_i||
#     1'den buyuk = en az bir dedektorun icinde. [0, 2] araliginda kirpiliyor.
#     Kumeler: Disarida, Sinirda, Icerde
# Girdi 2 - self uzakligi (g2): en yakin self ornege uzaklik / d_ref
#     d_ref = egitim saldirilarinin self'e uzakliklarinin medyani. [0, 2] araliginda.
#     Kumeler: Yakin, Orta, Uzak
# Cikti - anomali derecesi [0, 1]: Normal, Supheli, Saldiri
#
# Kural tablosu (9 kural):
#               Yakin     Orta      Uzak
#   Disarida    Normal    Normal    Supheli
#   Sinirda     Normal    Supheli   Saldiri
#   Icerde      Supheli   Saldiri   Saldiri
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- uyelik fonksiyonlari

def trimf(x, a, b, c):
    # ucgen: a'da 0, b'de 1, c'de 0. a == b veya b == c ise omuz gibi davranir
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-12), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / max(c - b, 1e-12), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)


def gaussmf(x, mean, sigma):
    return np.exp(-0.5 * ((np.asarray(x, float) - mean) / sigma) ** 2)


# her dilsel terim icin (ucgen parametreleri, gauss parametreleri)
# gauss sigma'lari ucgenle ayni yerde ~0.5 uyelik verecek sekilde secildi
INPUT1 = {  # kapsama, evren [0, 2], esik 1.0
    "Disarida": ((0.0, 0.0, 0.8), (0.0, 0.34)),
    "Sinirda":  ((0.6, 1.0, 1.4), (1.0, 0.17)),
    "Icerde":   ((1.2, 2.0, 2.0), (2.0, 0.34)),
}
INPUT2 = {  # self uzakligi, evren [0, 2], 1.0 = tipik saldiri uzakligi
    "Yakin": ((0.0, 0.0, 0.5), (0.0, 0.21)),
    "Orta":  ((0.2, 0.7, 1.2), (0.7, 0.21)),
    "Uzak":  ((0.9, 2.0, 2.0), (2.0, 0.47)),
}
OUTPUT = {  # anomali derecesi, evren [0, 1]
    "Normal":  ((0.0, 0.0, 0.4), (0.0, 0.17)),
    "Supheli": ((0.2, 0.5, 0.8), (0.5, 0.13)),
    "Saldiri": ((0.6, 1.0, 1.0), (1.0, 0.17)),
}
SUGENO_CONST = {"Normal": 0.0, "Supheli": 0.5, "Saldiri": 1.0}   # 0. derece Sugeno

RULES = [
    ("Disarida", "Yakin", "Normal"),
    ("Disarida", "Orta", "Normal"),
    ("Disarida", "Uzak", "Supheli"),
    ("Sinirda", "Yakin", "Normal"),
    ("Sinirda", "Orta", "Supheli"),
    ("Sinirda", "Uzak", "Saldiri"),
    ("Icerde", "Yakin", "Supheli"),
    ("Icerde", "Orta", "Saldiri"),
    ("Icerde", "Uzak", "Saldiri"),
]


def membership(x, params, mf):
    tri, gau = params
    return trimf(x, *tri) if mf == "tri" else gaussmf(x, *gau)


# ---- durulastirma (defuzzification)

def defuzzify(y, agg, method):
    # y: cikis evreni, agg: (n, len(y)) birlestirilmis cikti kumesi
    s = agg.sum(axis=1)
    empty = s < 1e-12
    if method == "centroid":
        out = (agg * y).sum(axis=1) / np.where(empty, 1, s)
    elif method == "bisector":
        # alani ikiye bolen nokta
        cum = np.cumsum(agg, axis=1)
        idx = np.argmax(cum >= cum[:, -1:] / 2, axis=1)
        out = y[idx]
    elif method == "mom":
        # en yuksek uyelige sahip noktalarin ortalamasi
        mx = agg.max(axis=1, keepdims=True)
        top = np.isclose(agg, mx) & (mx > 0)
        out = (top * y).sum(axis=1) / np.maximum(top.sum(axis=1), 1)
    else:
        raise ValueError(method)
    return np.where(empty, 0.5, out)   # hic kural atesmezse kararsiz -> 0.5


@dataclass
class FuzzyConfig:
    mf: str = "tri"              # "tri" veya "gauss"
    inference: str = "mamdani"   # "mamdani" veya "sugeno"
    defuzz: str = "centroid"     # "centroid", "bisector", "mom" (sadece mamdani)
    threshold: float = 0.5       # skor >= esik -> saldiri
    n_points: int = 201          # cikis evreni cozunurlugu


class FuzzyDetector:
    # NaIM (veya V-Detector) dedektorlerinin uzerine oturan bulanik karar katmani

    def __init__(self, detectors, self_space, d_ref, cfg=None):
        self.D = detectors
        self.ss = self_space
        self.d_ref = max(float(d_ref), 1e-6)
        self.cfg = cfg or FuzzyConfig()

    @classmethod
    def from_model(cls, model, X_nonself_train, cfg=None):
        # d_ref'i egitim saldirilarindan hesapla (testten bilgi sizmasin)
        d = model.ss.nearest_batch(X_nonself_train)
        return cls(model.D, model.ss, np.median(d), cfg)

    def inputs(self, X):
        g1 = np.clip(self.D.max_coverage_ratio(X), 0, 2)
        g2 = np.clip(self.ss.nearest_batch(X) / self.d_ref, 0, 2)
        return g1, g2

    def infer(self, g1, g2):
        # butun ara adimlari dondurur (demo.py bunu kullaniyor)
        cfg = self.cfg
        g1, g2 = np.atleast_1d(np.asarray(g1, float)), np.atleast_1d(np.asarray(g2, float))
        # 1) bulaniklastirma
        mu1 = {k: membership(g1, p, cfg.mf) for k, p in INPUT1.items()}
        mu2 = {k: membership(g2, p, cfg.mf) for k, p in INPUT2.items()}

        # 2) kural atesleme gucu: VE = min
        w = np.stack([np.minimum(mu1[a], mu2[b]) for a, b, _ in RULES], axis=1)
        out = {"mu1": mu1, "mu2": mu2, "w": w}

        if cfg.inference == "sugeno":
            # agirlikli ortalama
            z = np.array([SUGENO_CONST[c] for _, _, c in RULES])
            s = w.sum(axis=1)
            out["score"] = np.where(s > 1e-12, (w * z).sum(axis=1) / np.where(s > 0, s, 1), 0.5)
            return out

        # 3) mamdani: her kural cikti kumesini kirpar (min), sonra max ile birlestir
        y = np.linspace(0, 1, cfg.n_points)
        out_mf = {k: membership(y, p, cfg.mf) for k, p in OUTPUT.items()}
        agg = np.zeros((len(g1), len(y)))
        for j, (_, _, c) in enumerate(RULES):
            agg = np.maximum(agg, np.minimum(w[:, j:j + 1], out_mf[c][None, :]))

        # 4) durulastirma
        out.update(y=y, agg=agg, score=defuzzify(y, agg, cfg.defuzz))
        return out

    def score_from_inputs(self, g1, g2):
        return self.infer(g1, g2)["score"]

    def score(self, X):
        return self.score_from_inputs(*self.inputs(X))

    def predict(self, X):
        return (self.score(X) >= self.cfg.threshold).astype(int)
