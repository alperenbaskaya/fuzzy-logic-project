# V-Detector (Ji & Dasgupta, 2004) - baseline.
# Makale Tablo 7-8'de bununla karsilastiriyor.
#
# 1) [0,1]^d icinde rastgele nokta at
# 2) zaten bir dedektorun icindeyse sayaci arttir, yeterince arka arkaya
#    kapsanmissa dur (uzay doldu)
# 3) degilse r = d_min(x, self) - r_self, r > 0 ise dedektor olarak ekle
# Durma kurali orijinalden biraz basit (ASSUMPTIONS A5).
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detectors import DetectorSet, SelfSpace


@dataclass
class VDetectorConfig:
    r_self: float = 0.05
    c0: float = 0.99              # hedef kapsama
    max_detectors: int = 5000
    max_trials: int = 500_000


@dataclass
class TrainInfo:
    n_detectors: int = 0
    train_time: float = 0.0
    n_dist: int = 0
    extra: dict = field(default_factory=dict)


def train_vdetector(X_self, cfg, rng):
    t0 = time.perf_counter()
    dim = X_self.shape[1]
    ss = SelfSpace(X_self)                      # burada FCM yok
    D = DetectorSet(dim)
    need = int(np.ceil(1.0 / (1.0 - cfg.c0)))   # c0=0.99 -> 100 aday
    streak = 0

    for _ in range(cfg.max_trials):
        if len(D) >= cfg.max_detectors:
            break
        x = rng.random(dim)
        if D.covers_point(x):
            streak += 1
            if streak >= need:
                break
            continue
        r = ss.nearest(x) - cfg.r_self
        if r > 0:
            D.add(x, r)
            streak = 0

    return D, TrainInfo(len(D), time.perf_counter() - t0, ss.n_dist)
