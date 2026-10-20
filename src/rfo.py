"""Red Fox Optimization (RFO) -- en büyükleme sürümü.

D. Połap ve M. Woźniak, "Red fox optimization algorithm", Expert Systems with
Applications, 2021. NaIM-IDS'te sınır dedektörlerinin konum ve yarıçapını
optimize etmek için kullanılır (makale Bölüm 3.3.1, Algoritma 1).

Her iterasyonda üç aşama:
  1) Küresel arama: her tilki en iyi tilkiye doğru rastgele bir adım atar
       x' = x + alpha * sign(x_best - x),  alpha ~ U(0, ||x_best - x||)
  2) Yerel arama ("avı fark etme"): mu ~ U(0,1); mu > 0.75 ise tilki yerinde kalır,
     değilse rastgele açılarla küçük bir dönüş hareketi yapar.
  3) Üreme: en kötü %5'lik tilkiler, en iyi iki tilkinin "yaşam alanı" içinde
     yeniden üretilir.
Her adımda açgözlü seçim yapılır: yeni konum daha iyiyse kabul edilir.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _local_move(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RFO yerel arama hareketi (orijinal makaledeki açısal formül)."""
    d = len(x)
    a = rng.uniform(0.0, 0.2)
    phi = rng.uniform(0.0, 2 * np.pi, size=d)
    rr = a * np.sin(phi[0]) / phi[0] if phi[0] != 0 else rng.random()
    # x'_k = x_k + a*rr*( sum_{j=1..k} sin(phi_j) + cos(phi_{k+1}) ),  son boyutta cos terimi yok
    sins = np.concatenate([[0.0], np.cumsum(np.sin(phi[1:d]))])
    coss = np.concatenate([np.cos(phi[1:d]), [0.0]])
    return x + a * rr * (sins + coss)


def rfo_maximize(
    fitness: Callable[[np.ndarray], float],
    init_pop: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    pop = np.clip(init_pop.astype(float), lb, ub)
    fit = np.array([fitness(p) for p in pop])
    n = len(pop)

    def try_move(i: int, cand: np.ndarray) -> None:
        cand = np.clip(cand, lb, ub)
        f = fitness(cand)
        if f > fit[i]:
            pop[i], fit[i] = cand, f

    for _ in range(n_iter):
        order = np.argsort(-fit)
        pop, fit = pop[order], fit[order]
        best = pop[0].copy()

        # 1) küresel arama
        for i in range(1, n):
            dist = np.linalg.norm(best - pop[i])
            if dist > 0:
                try_move(i, pop[i] + rng.uniform(0, dist) * np.sign(best - pop[i]))

        # 2) yerel arama
        for i in range(n):
            if rng.random() <= 0.75:
                try_move(i, _local_move(pop[i], rng))

        # 3) üreme: en kötü %5 yaşam alanı içinde yenilenir
        order = np.argsort(-fit)
        pop, fit = pop[order], fit[order]
        if n >= 3:
            center = (pop[0] + pop[1]) / 2
            diam = max(np.linalg.norm(pop[0] - pop[1]), 1e-3)
            for i in range(n - max(1, int(0.05 * n)), n):
                if rng.random() >= 0.45:
                    new = rng.uniform(lb, ub)
                else:
                    new = center + rng.uniform(-0.5, 0.5, size=len(center)) * diam
                new = np.clip(new, lb, ub)
                pop[i], fit[i] = new, fitness(new)

    i = int(np.argmax(fit))
    return pop[i].copy(), float(fit[i])
