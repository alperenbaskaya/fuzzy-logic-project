## E4/E5 ozet (3 sinif x 20 tekrar, %50 egitim, ayrik test)

| Yontem | DR % | FAR % | F1 % | AUC % |
|---|---|---|---|---|
| Keskin (makale) | 94.8 ± 5.3 | 5.0 ± 5.4 | 96.1 ± 3.3 | 94.9 ± 4.1 |
| Kapsama skoru (FIS yok) | (= keskin) | (= keskin) | (= keskin) | 98.5 ± 1.8 |
| FIS ucgen mamdani centroid | 83.1 ± 16.6 | 0.9 ± 2.9 | 89.6 ± 10.6 | 96.7 ± 4.2 |
| FIS gauss mamdani centroid | 82.6 ± 16.6 | 0.6 ± 1.9 | 89.4 ± 10.7 | 97.0 ± 3.8 |
| FIS ucgen mamdani bisector | 83.5 ± 16.6 | 1.1 ± 3.8 | 89.8 ± 10.6 | 96.3 ± 4.9 |
| FIS ucgen mamdani mom | 88.1 ± 14.0 | 5.3 ± 10.7 | 91.8 ± 8.5 | 94.1 ± 8.2 |
| FIS ucgen sugeno | 83.3 ± 16.6 | 0.9 ± 2.9 | 89.7 ± 10.7 | 97.5 ± 2.9 |
| FIS gauss sugeno | 82.9 ± 16.6 | 0.6 ± 1.9 | 89.6 ± 10.7 | 98.5 ± 1.7 |

## Sinif bazinda

|                                                   |    DR |   FAR |    F1 |   AUC |
|:--------------------------------------------------|------:|------:|------:|------:|
| ('iris-setosa', 'FIS ucgen mamdani centroid')     | 100   |   0   | 100   | 100   |
| ('iris-setosa', 'Keskin (makale)')                |  99.6 |   1.2 |  99.5 |  99.2 |
| ('iris-versicolor', 'FIS ucgen mamdani centroid') |  80.9 |   0.2 |  89.3 |  97.7 |
| ('iris-versicolor', 'Keskin (makale)')            |  91.7 |   6   |  94.1 |  92.8 |
| ('iris-virginica', 'FIS ucgen mamdani centroid')  |  68.3 |   2.4 |  79.5 |  92.3 |
| ('iris-virginica', 'Keskin (makale)')             |  93.2 |   7.8 |  94.5 |  92.7 |

## Wilcoxon isaretli sira testi (n = 60)

| Karsilastirma | Metrik | Ort. fark (bulanik - keskin) | p-degeri |
|---|---|---|---|
| FIS ucgen mamdani centroid vs keskin | DR | -11.77 | 0.0000 |
| FIS ucgen mamdani centroid vs keskin | FAR | -4.13 | 0.0000 |
| FIS ucgen mamdani centroid vs keskin | F1 | -6.46 | 0.0000 |
| FIS ucgen sugeno vs keskin | DR | -11.53 | 0.0000 |
| FIS ucgen sugeno vs keskin | FAR | -4.13 | 0.0000 |
| FIS ucgen sugeno vs keskin | F1 | -6.33 | 0.0000 |

## E4b esik validasyonla ayarlandiginda (egitim 70/30)

| Yontem | DR % | FAR % | F1 % |
|---|---|---|---|
| Keskin (makale) | 95.3 ± 5.9 | 6.1 ± 6.4 | 96.0 ± 3.8 |
| Kapsama skoru + ayarli esik | 91.6 ± 7.5 | 4.5 ± 6.2 | 94.4 ± 4.5 |
| FIS ucgen mamdani centroid + ayarli esik | 90.1 ± 9.7 | 8.3 ± 11.9 | 92.6 ± 6.0 |
| FIS ucgen sugeno + ayarli esik | 91.6 ± 10.1 | 7.9 ± 11.3 | 93.5 ± 6.3 |

| Karsilastirma | Metrik | Ort. fark | p-degeri |
|---|---|---|---|
| Kapsama skoru + ayarli esik vs keskin | DR | -3.63 | 0.0006 |
| Kapsama skoru + ayarli esik vs keskin | FAR | -1.67 | 0.0230 |
| Kapsama skoru + ayarli esik vs keskin | F1 | -1.61 | 0.0045 |
| FIS ucgen mamdani centroid + ayarli esik vs keskin | DR | -5.13 | 0.0001 |
| FIS ucgen mamdani centroid + ayarli esik vs keskin | FAR | +2.20 | 0.3883 |
| FIS ucgen mamdani centroid + ayarli esik vs keskin | F1 | -3.43 | 0.0000 |
| FIS ucgen sugeno + ayarli esik vs keskin | DR | -3.63 | 0.0026 |
| FIS ucgen sugeno + ayarli esik vs keskin | FAR | +1.80 | 0.4725 |
| FIS ucgen sugeno + ayarli esik vs keskin | F1 | -2.54 | 0.0004 |

## E7 gurultu (ortalama)

|   sigma |   ('DR', 'Bulanik') |   ('DR', 'Keskin') |   ('FAR', 'Bulanik') |   ('FAR', 'Keskin') |   ('F1', 'Bulanik') |   ('F1', 'Keskin') |
|--------:|--------------------:|-------------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|    0    |                83.1 |               94.8 |                  0.9 |                 5   |                89.6 |               96.1 |
|    0.02 |                83.6 |               94.9 |                  1.1 |                 4.9 |                89.9 |               96.1 |
|    0.05 |                83.9 |               93.7 |                  1.5 |                 6.9 |                90   |               95   |
|    0.08 |                84.9 |               92.8 |                  3.7 |                10.5 |                90.1 |               93.6 |
|    0.12 |                86.1 |               90.4 |                  8.5 |                14.6 |                89.9 |               91.3 |
