# NaIM-IDS dedektor uretimi (makale Bolum 3.2 - 3.3)
#
# 1) self ornekleri FCM ile kumele, c'yi Com/Sep ile sec (m = 2.3 sabit)
# 2) egitim saldirilarini self'e uzakligina gore ikiye ayir:
#      self'e en yakin %30 -> sinir bolgesi, gerisi -> sinir disi (ASSUMPTIONS A6)
# 3) sinir bolgesi: RFO ile dedektor konumu/yaricapi ara (Alg. 1), %95 kapsanana kadar
# 4) sinir disi: Denk. 20 ile alt populasyonlara bol, her birinde GA calistir
#
# Dedektor fitness'i (ASSUMPTIONS A7):
#   F = kapsama + 0.5 * r / sqrt(d) - 0.3 * ortusme   (r <= 0 ise gecersiz)
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .detectors import DetectorSet, SelfSpace
from .fcm import fcm, select_fcm
from .ga import ga_maximize
from .rfo import rfo_maximize
from .vdetector import TrainInfo


@dataclass
class NaIMConfig:
    r_self: float = 0.05
    use_fcm: bool = True            # False -> FCM yok (ablasyon icin)
    fcm_c: int | None = None        # None -> Com/Sep ile sec
    fcm_m: float = 2.3              # makale Sekil 4b
    boundary_q: float = 0.3
    b_threshold: float = 0.15       # Denk. 20 esigi
    target_cov: float = 0.95        # Alg. 1
    max_detectors: int = 1000
    max_fail: int = 30
    span: float = 0.2               # capa etrafindaki arama kutusu
    rfo_pop: int = 20
    rfo_iter: int = 15
    ga_pop: int = 20
    ga_gen: int = 15
    w_r: float = 0.5
    w_ov: float = 0.3


def partition_by_difference(X: np.ndarray, b: float) -> list[np.ndarray]:
    # Denk. 20: ornek en yakin gruba girer, fark b'den buyukse yeni grup acar
    centers: list[np.ndarray] = []
    groups: list[list[int]] = []
    for i, p in enumerate(X):
        if centers:
            diff = np.array([np.mean(np.abs(p - c)) for c in centers])
            k = int(np.argmin(diff))
            if diff[k] <= b:
                groups[k].append(i)
                centers[k] = X[groups[k]].mean(axis=0)
                continue
        centers.append(p.copy())
        groups.append([i])
    return [np.array(g) for g in groups]


class NaIM:
    def __init__(self, cfg: NaIMConfig):
        self.cfg = cfg
        self.D: DetectorSet | None = None
        self.ss: SelfSpace | None = None
        self.fcm_res = None
        self.info = TrainInfo()

    # ---- fitness
    def _fitness_factory(self, targets: np.ndarray, n_total: int):
        cfg, ss, D = self.cfg, self.ss, self.D
        sqrt_d = np.sqrt(targets.shape[1])

        def fitness(c: np.ndarray) -> float:
            r = ss.nearest(c) - cfg.r_self
            if r <= 0:
                return r - 1.0      # self ile cakisiyor, gecersiz
            cov = np.count_nonzero(np.linalg.norm(targets - c, axis=1) <= r) / max(1, n_total)
            ov = 0.0
            if len(D):
                dd = np.linalg.norm(D.C - c, axis=1)
                ov = float(np.max(np.clip((r + D.R - dd) / (2 * np.minimum(r, D.R)), 0, 1)))
            return cov + cfg.w_r * r / sqrt_d - cfg.w_ov * ov
        return fitness

    def _grow_region(self, region_idx: np.ndarray, Xn: np.ndarray, covered: np.ndarray,
                     optimizer: str, rng: np.random.Generator) -> int:
        # bolgedeki saldirilarin target_cov kadari kapsanana kadar dedektor ekle
        cfg, d = self.cfg, Xn.shape[1]
        added, fails = 0, 0
        R = Xn[region_idx]
        while (covered[region_idx].mean() < cfg.target_cov and len(self.D) < cfg.max_detectors
               and fails < cfg.max_fail):
            unc = region_idx[~covered[region_idx]]
            anchor = Xn[rng.choice(unc)]
            fitness = self._fitness_factory(Xn[unc], len(region_idx))

            if optimizer == "rfo":
                lb, ub = np.clip(anchor - cfg.span, 0, 1), np.clip(anchor + cfg.span, 0, 1)
                init = anchor + rng.normal(0, cfg.span / 3, (cfg.rfo_pop, d))
                init[0] = anchor    # capa noktasi her zaman gecerli
                best, _ = rfo_maximize(fitness, init, lb, ub, cfg.rfo_iter, rng)
            else:
                lb = np.clip(R.min(axis=0) - cfg.span / 2, 0, 1)
                ub = np.clip(R.max(axis=0) + cfg.span / 2, 0, 1)
                pick = rng.choice(unc, size=cfg.ga_pop)
                init = Xn[pick] + rng.normal(0, 0.02, (cfg.ga_pop, d))
                init[0] = anchor
                best, _ = ga_maximize(fitness, init, lb, ub, cfg.ga_gen, rng)

            r = self.ss.nearest(best) - cfg.r_self
            inside = np.linalg.norm(Xn - best, axis=1) <= r
            if r > 0 and np.any(inside[unc]):
                self.D.add(best, r)
                covered |= inside
                added += 1
                fails = 0
            else:
                fails += 1
        return added

    # ---- egitim
    def fit(self, X_self: np.ndarray, X_nonself: np.ndarray,
            rng: np.random.Generator) -> "NaIM":
        cfg = self.cfg
        t0 = time.perf_counter()
        d = X_self.shape[1]

        # 1) self orneklerini FCM ile kumele
        if cfg.use_fcm:
            if cfg.fcm_c is None:
                # m'yi de aratinca hep en buyuk m'yi seciyor (ASSUMPTIONS A3), o yuzden m sabit
                c_max = max(2, min(10, len(X_self) // 3))
                self.fcm_res, _ = select_fcm(X_self, range(2, c_max + 1),
                                             m_values=(cfg.fcm_m,), rng=rng)
            else:
                self.fcm_res = fcm(X_self, cfg.fcm_c, cfg.fcm_m, rng=rng)
            self.ss = SelfSpace(X_self, self.fcm_res.centers, self.fcm_res.labels)
        else:
            self.ss = SelfSpace(X_self)

        # 2) sinir / sinir disi ayrimi
        dns = self.ss.nearest_batch(X_nonself)
        Xn = X_nonself[dns > cfg.r_self]    # self kuresinin icindekiler zaten yakalanamaz
        dns = dns[dns > cfg.r_self]
        thr = np.quantile(dns, cfg.boundary_q)
        boundary = np.flatnonzero(dns <= thr)
        nonboundary = np.flatnonzero(dns > thr)

        self.D = DetectorSet(d)
        covered = np.zeros(len(Xn), bool)

        # 3) sinir dedektorleri (RFO)
        n_b = self._grow_region(boundary, Xn, covered, "rfo", rng)

        # 4) sinir disi dedektorler (alt populasyon + GA)
        groups = partition_by_difference(Xn[nonboundary], cfg.b_threshold) if len(nonboundary) else []
        n_nb = 0
        for g in groups:
            n_nb += self._grow_region(nonboundary[g], Xn, covered, "ga", rng)

        self.info = TrainInfo(
            n_detectors=len(self.D),
            train_time=time.perf_counter() - t0,
            n_dist=self.ss.n_dist,
            extra={
                "n_boundary_det": n_b, "n_nonboundary_det": n_nb,
                "n_subpop": len(groups), "train_coverage": float(covered.mean()),
                "fcm_c": None if self.fcm_res is None else len(self.fcm_res.centers),
                "fcm_m": None if self.fcm_res is None else self.fcm_res.m,
            },
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.D.predict(X)
