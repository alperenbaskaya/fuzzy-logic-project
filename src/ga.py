"""İkili kodlu genetik algoritma (en büyükleme).

Makale Bölüm 3.3.2: sınır dışı dedektörler alt popülasyonlar içinde GA ile evrilir.
  - Seçim: rulet tekerleği
  - Çaprazlama: uniform çaprazlama
  - Mutasyon: hızlı bit mutasyonu ("fast mutation", makaledeki kaynak [46])
    Çevrilecek bit sayısı k, kuvvet yasası dağılımından seçilir: P(k) ~ k^(-beta).
    Çoğu zaman az bit, ara sıra çok bit çevrilir; bu yerel optimumdan kaçmayı sağlar.
  - Elitizm: en iyi birey bir sonraki nesle aynen aktarılır.

Her gerçek değerli boyut `bits` bit ile [lb, ub] aralığına kodlanır.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def encode(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits - 1
    q = np.round((x - lb) / np.fmax(ub - lb, 1e-12) * levels).astype(int)
    q = np.clip(q, 0, levels)
    return ((q[:, None] >> np.arange(bits - 1, -1, -1)) & 1).ravel().astype(np.uint8)


def decode(g: np.ndarray, lb: np.ndarray, ub: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits - 1
    q = g.reshape(-1, bits) @ (1 << np.arange(bits - 1, -1, -1))
    return lb + q / levels * (ub - lb)


def _fast_mutation(g: np.ndarray, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    L = len(g)
    ks = np.arange(1, max(2, L // 2) + 1)
    p = ks ** (-beta)
    k = rng.choice(ks, p=p / p.sum())
    flip = rng.random(L) < k / L
    out = g.copy()
    out[flip] ^= 1
    return out


def ga_maximize(
    fitness: Callable[[np.ndarray], float],
    init_pop: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_gen: int,
    rng: np.random.Generator,
    bits: int = 12,
    pc: float = 0.9,
) -> tuple[np.ndarray, float]:
    G = np.array([encode(np.clip(x, lb, ub), lb, ub, bits) for x in init_pop])
    n = len(G)
    fit = np.array([fitness(decode(g, lb, ub, bits)) for g in G])

    for _ in range(n_gen):
        elite = G[np.argmax(fit)].copy()
        # rulet: uygunluklar pozitife kaydırılır
        w = fit - fit.min() + 1e-9
        parents = G[rng.choice(n, size=n, p=w / w.sum())]
        children = parents.copy()
        for i in range(0, n - 1, 2):
            if rng.random() < pc:                      # uniform çaprazlama
                mask = rng.random(G.shape[1]) < 0.5
                children[i, mask], children[i + 1, mask] = parents[i + 1, mask], parents[i, mask]
        children = np.array([_fast_mutation(c, rng) for c in children])
        children[0] = elite                            # elitizm
        G = children
        fit = np.array([fitness(decode(g, lb, ub, bits)) for g in G])

    i = int(np.argmax(fit))
    return decode(G[i], lb, ub, bits), float(fit[i])
