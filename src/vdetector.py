"""V-Detector: değişken yarıçaplı gerçek değerli negatif seçim (baseline).

Z. Ji ve D. Dasgupta, GECCO 2004 (makaledeki kaynak [52]). Makale Tablo 7 ve 8'de
NaIM-IDS ile karşılaştırılan klasik yapay bağışıklık yöntemidir.

Algoritma (yalnızca self örnekleri kullanır):
  1) [0,1]^d içinde rastgele bir aday nokta x üret.
  2) x mevcut bir dedektörün içindeyse sayaç t'yi artır; t >= 1/(1-c0) olunca
     non-self uzayın yaklaşık c0 oranında kapsandığı kabul edilir ve durulur.
  3) Değilse r = d_min(x, self) - r_self hesapla; r > 0 ise (x, r) dedektörünü ekle
     ve t'yi sıfırla.
  4) Dedektör sayısı sınırına ulaşınca dur.

Not: Orijinal makaledeki hipotez testi tabanlı durma kuralı yerine basitleştirilmiş
ardışık kapsama sayacı kullanılmıştır (varsayım A5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detectors import DetectorSet, SelfSpace


@dataclass
class VDetectorConfig:
    r_self: float = 0.05        # self örnek yarıçapı
    c0: float = 0.99            # hedef kapsama oranı
    max_detectors: int = 5000
    max_trials: int = 500_000


@dataclass
class TrainInfo:
    n_detectors: int = 0
    train_time: float = 0.0
    n_dist: int = 0             # tolerans aşamasındaki uzaklık hesabı sayısı
    extra: dict = field(default_factory=dict)


def train_vdetector(X_self: np.ndarray, cfg: VDetectorConfig,
                    rng: np.random.Generator) -> tuple[DetectorSet, TrainInfo]:
    t0 = time.perf_counter()
    dim = X_self.shape[1]
    ss = SelfSpace(X_self)                       # kaba kuvvet tolerans
    D = DetectorSet(dim)
    need = int(np.ceil(1.0 / (1.0 - cfg.c0)))    # ardışık kapsanan aday sayısı
    covered_streak = 0

    for _ in range(cfg.max_trials):
        if len(D) >= cfg.max_detectors:
            break
        x = rng.random(dim)
        if D.covers_point(x):
            covered_streak += 1
            if covered_streak >= need:
                break
            continue
        r = ss.nearest(x) - cfg.r_self
        if r > 0:
            D.add(x, r)
            covered_streak = 0

    info = TrainInfo(len(D), time.perf_counter() - t0, ss.n_dist)
    return D, info
