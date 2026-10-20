"""NaIM-IDS uyarlanabilir dedektör üretimi (makale Bölüm 3.2 - 3.3).

Akış:
  1) Self örnekler FCM ile kümelenir; (c, m) Com/Sep indeksiyle seçilir (Bölüm 3.2).
     Tolerans hesabı (d_min(x, self)) küme merkezleri ve küme boyutlarıyla budanır.
  2) Eğitim non-self örnekleri, self'e uzaklıklarına göre ikiye ayrılır:
       sınır bölgesi     : self'e en yakın %q'luk dilim
       sınır dışı bölge  : geri kalanlar
     (makale bu ayrımın ölçütünü vermez -> varsayım A6)
  3) Sınır dedektörleri (Bölüm 3.3.1, Algoritma 1): kapsanmamış bir sınır örneği
     çapa olarak seçilir, çevresinde RFO ile en iyi (merkez, yarıçap) aranır.
     Sınır örneklerinin %95'i kapsanana kadar tekrarlanır.
  4) Sınır dışı dedektörler (Bölüm 3.3.2): sınır dışı non-self örnekler fark eşiği b'ye
     göre alt popülasyonlara bölünür (Denk. 20); her alt popülasyon içinde GA ile
     dedektörler evrilir.

Dedektör uygunluğu (çok amaçlı, makale kaynak [45]; ağırlıklar makalede verilmez
-> varsayım A7):
  F(c) = kapsama  +  w_r * r / sqrt(d)  -  w_ov * örtüşme
    kapsama : yarıçap r içine giren, henüz kapsanmamış hedef örneklerin oranı
    r       : d_min(c, self) - r_self   (self ile eşleşmeme koşulu: r > 0)
    örtüşme : mevcut dedektörlerle en büyük göreli örtüşme, [0, 1]
"""
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
    use_fcm: bool = True            # False -> kaba kuvvet tolerans (ablasyon)
    fcm_c: int | None = None        # None -> Com/Sep ile otomatik seçim
    fcm_m: float = 2.3              # makale Şekil 4b: en iyi bulanıklık katsayısı
    boundary_q: float = 0.3         # self'e en yakın %30 non-self = sınır bölgesi
    b_threshold: float = 0.15       # alt popülasyon fark eşiği (Denk. 20)
    target_cov: float = 0.95        # Algoritma 1: rate <= 0.95 olduğu sürece devam
    max_detectors: int = 1000
    max_fail: int = 30
    span: float = 0.2               # çapa çevresindeki arama kutusunun yarı genişliği
    rfo_pop: int = 20
    rfo_iter: int = 15
    ga_pop: int = 20
    ga_gen: int = 15
    w_r: float = 0.5
    w_ov: float = 0.3


def partition_by_difference(X: np.ndarray, b: float) -> list[np.ndarray]:
    """Denk. 20: Diff(p, C) = |p - c|. Lider tabanlı alt popülasyon bölmesi.

    Her örnek, ortalama mutlak farkı en küçük olan sınıfa eklenir; bu fark eşik b'den
    büyükse örnek yeni bir sınıf başlatır. Sınıf merkezi her eklemede güncellenir.
    Döndürür: her alt popülasyon için X içindeki indeks dizisi.
    """
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

    # ------------------------------------------------------------------ uygunluk
    def _fitness_factory(self, targets: np.ndarray, n_total: int):
        cfg, ss, D = self.cfg, self.ss, self.D
        sqrt_d = np.sqrt(targets.shape[1])

        def fitness(c: np.ndarray) -> float:
            r = ss.nearest(c) - cfg.r_self
            if r <= 0:
                return r - 1.0                          # geçersiz: self ile eşleşiyor
            cov = np.count_nonzero(np.linalg.norm(targets - c, axis=1) <= r) / max(1, n_total)
            ov = 0.0
            if len(D):
                dd = np.linalg.norm(D.C - c, axis=1)
                ov = float(np.max(np.clip((r + D.R - dd) / (2 * np.minimum(r, D.R)), 0, 1)))
            return cov + cfg.w_r * r / sqrt_d - cfg.w_ov * ov
        return fitness

    def _grow_region(self, region_idx: np.ndarray, Xn: np.ndarray, covered: np.ndarray,
                     optimizer: str, rng: np.random.Generator) -> int:
        """Bir bölgedeki non-self örneklerin target_cov oranı kapsanana kadar dedektör ekler."""
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
                init[0] = anchor                         # çapa her zaman geçerli bir aday
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

    # ------------------------------------------------------------------ eğitim
    def fit(self, X_self: np.ndarray, X_nonself: np.ndarray,
            rng: np.random.Generator) -> "NaIM":
        cfg = self.cfg
        t0 = time.perf_counter()
        d = X_self.shape[1]

        # 1) self örneklerin bulanık kümelenmesi
        if cfg.use_fcm:
            if cfg.fcm_c is None:
                # c, Com/Sep ile seçilir; m makalenin Şekil 4b'deki en iyi değerinde sabit
                # tutulur, çünkü Com terimi u^m içerdiği için indeks her zaman en büyük
                # m'yi seçme eğilimindedir (varsayım A3).
                c_max = max(2, min(10, len(X_self) // 3))
                self.fcm_res, _ = select_fcm(X_self, range(2, c_max + 1),
                                             m_values=(cfg.fcm_m,), rng=rng)
            else:
                self.fcm_res = fcm(X_self, cfg.fcm_c, cfg.fcm_m, rng=rng)
            self.ss = SelfSpace(X_self, self.fcm_res.centers, self.fcm_res.labels)
        else:
            self.ss = SelfSpace(X_self)

        # 2) sınır / sınır dışı ayrımı
        dns = self.ss.nearest_batch(X_nonself)
        Xn = X_nonself[dns > cfg.r_self]                # self küresi içindekiler tespit edilemez
        dns = dns[dns > cfg.r_self]
        thr = np.quantile(dns, cfg.boundary_q)
        boundary = np.flatnonzero(dns <= thr)
        nonboundary = np.flatnonzero(dns > thr)

        self.D = DetectorSet(d)
        covered = np.zeros(len(Xn), bool)

        # 3) RFO ile sınır dedektörleri
        n_b = self._grow_region(boundary, Xn, covered, "rfo", rng)

        # 4) alt popülasyonlar + GA ile sınır dışı dedektörler
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
