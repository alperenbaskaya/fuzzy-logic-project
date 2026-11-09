# Red Fox Optimization (Polap & Wozniak, 2021) - maximize versiyonu.
# Sinir dedektorlerinin merkezini bulmak icin kullaniyorum (makale Alg. 1).
#
# Her iterasyonda:
#  1) global arama: tilkiler en iyi tilkiye dogru rastgele adim atar
#  2) yerel arama: %75 ihtimalle etrafta kucuk bir donus hareketi
#  3) ureme: en kotu %5 yeniden uretilir
# Her adimda yeni konum daha iyiyse kabul ediliyor (greedy).
from __future__ import annotations

import numpy as np


def _local_move(x, rng):
    # orijinal makaledeki acili hareket formulu
    d = len(x)
    a = rng.uniform(0.0, 0.2)
    phi = rng.uniform(0.0, 2 * np.pi, size=d)
    rr = a * np.sin(phi[0]) / phi[0] if phi[0] != 0 else rng.random()
    sins = np.concatenate([[0.0], np.cumsum(np.sin(phi[1:d]))])
    coss = np.concatenate([np.cos(phi[1:d]), [0.0]])
    return x + a * rr * (sins + coss)


def rfo_maximize(fitness, init_pop, lb, ub, n_iter, rng):
    pop = np.clip(init_pop.astype(float), lb, ub)
    fit = np.array([fitness(p) for p in pop])
    n = len(pop)

    def try_move(i, cand):
        cand = np.clip(cand, lb, ub)
        f = fitness(cand)
        if f > fit[i]:
            pop[i], fit[i] = cand, f

    for _ in range(n_iter):
        order = np.argsort(-fit)
        pop, fit = pop[order], fit[order]
        best = pop[0].copy()

        # 1) global arama
        for i in range(1, n):
            dist = np.linalg.norm(best - pop[i])
            if dist > 0:
                try_move(i, pop[i] + rng.uniform(0, dist) * np.sign(best - pop[i]))

        # 2) yerel arama
        for i in range(n):
            if rng.random() <= 0.75:
                try_move(i, _local_move(pop[i], rng))

        # 3) ureme (orijinal formulu degistirdim, ASSUMPTIONS A9)
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
