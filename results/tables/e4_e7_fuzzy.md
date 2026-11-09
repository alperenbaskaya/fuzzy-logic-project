## E4/E5 bulanik katman (10 tekrar)

### bot-iot-clean

| Yontem | DR % | FAR % | F1 % | AUC % |
|---|---|---|---|---|
| Keskin (makale) | 98.60 ± 0.61 | 1.71 ± 1.57 | 99.22 ± 0.30 | 98.45 ± 0.78 |
| Kapsama skoru + ayarli esik | 99.00 ± 0.25 | 2.61 ± 2.05 | 99.39 ± 0.10 | 99.26 ± 0.50 |
| FIS ucgen mamdani centroid (esik 0.5) | 95.21 ± 0.96 | 0.71 ± 0.75 | 97.51 ± 0.50 | 99.08 ± 0.33 |
| FIS ucgen mamdani centroid + ayarli esik | 98.56 ± 0.34 | 5.26 ± 2.60 | 99.05 ± 0.18 | 99.08 ± 0.33 |
| FIS gauss mamdani centroid (esik 0.5) | 95.29 ± 1.09 | 0.62 ± 0.78 | 97.56 ± 0.56 | 99.60 ± 0.26 |
| FIS gauss mamdani centroid + ayarli esik | 99.09 ± 0.53 | 6.30 ± 2.45 | 99.27 ± 0.24 | 99.60 ± 0.26 |
| FIS ucgen mamdani bisector (esik 0.5) | 95.35 ± 1.10 | 0.71 ± 0.75 | 97.59 ± 0.57 | 99.01 ± 0.35 |
| FIS ucgen mamdani bisector + ayarli esik | 98.37 ± 0.38 | 4.27 ± 2.31 | 99.00 ± 0.17 | 99.01 ± 0.35 |
| FIS ucgen mamdani mom (esik 0.5) | 96.35 ± 0.92 | 1.09 ± 1.10 | 98.09 ± 0.45 | 98.58 ± 0.54 |
| FIS ucgen mamdani mom + ayarli esik | 97.24 ± 0.90 | 6.40 ± 7.13 | 98.33 ± 0.48 | 98.58 ± 0.54 |
| FIS ucgen sugeno (esik 0.5) | 95.34 ± 1.13 | 0.71 ± 0.75 | 97.58 ± 0.58 | 99.17 ± 0.32 |
| FIS ucgen sugeno + ayarli esik | 98.63 ± 0.32 | 4.45 ± 1.21 | 99.12 ± 0.18 | 99.17 ± 0.32 |
| FIS gauss sugeno (esik 0.5) | 95.33 ± 1.07 | 0.62 ± 0.78 | 97.58 ± 0.55 | 99.78 ± 0.18 |
| FIS gauss sugeno + ayarli esik | 99.63 ± 0.42 | 7.20 ± 2.50 | 99.51 ± 0.15 | 99.78 ± 0.18 |

| Karsilastirma (vs keskin) | Metrik | Ort. fark | p |
|---|---|---|---|
| FIS ucgen mamdani centroid (esik 0.5) | DR | -3.39 | 0.0020 |
| FIS ucgen mamdani centroid (esik 0.5) | FAR | -1.00 | 0.0156 |
| FIS ucgen mamdani centroid (esik 0.5) | F1 | -1.71 | 0.0020 |
| FIS ucgen mamdani centroid + ayarli esik | DR | -0.04 | 0.6758 |
| FIS ucgen mamdani centroid + ayarli esik | FAR | +3.55 | 0.0020 |
| FIS ucgen mamdani centroid + ayarli esik | F1 | -0.17 | 0.0840 |
| FIS gauss sugeno + ayarli esik | DR | +1.03 | 0.0059 |
| FIS gauss sugeno + ayarli esik | FAR | +5.50 | 0.0020 |
| FIS gauss sugeno + ayarli esik | F1 | +0.29 | 0.0195 |
| Kapsama skoru + ayarli esik | DR | +0.40 | 0.0078 |
| Kapsama skoru + ayarli esik | FAR | +0.90 | 0.0078 |
| Kapsama skoru + ayarli esik | F1 | +0.17 | 0.0391 |

| Karsilastirma (vs kapsama skoru + ayarli esik) | Metrik | Ort. fark | p |
|---|---|---|---|
| FIS ucgen mamdani centroid + ayarli esik | F1 | -0.34 | 0.0020 |
| FIS ucgen mamdani centroid + ayarli esik | FAR | +2.65 | 0.0234 |
| FIS ucgen mamdani centroid + ayarli esik | AUC | -0.18 | 0.3223 |
| FIS gauss sugeno + ayarli esik | F1 | +0.12 | 0.0840 |
| FIS gauss sugeno + ayarli esik | FAR | +4.60 | 0.0020 |
| FIS gauss sugeno + ayarli esik | AUC | +0.51 | 0.0020 |

### skin

| Yontem | DR % | FAR % | F1 % | AUC % |
|---|---|---|---|---|
| Keskin (makale) | 98.49 ± 0.44 | 0.01 ± 0.02 | 99.24 ± 0.22 | 99.24 ± 0.23 |
| Kapsama skoru + ayarli esik | 99.19 ± 0.33 | 0.23 ± 0.16 | 99.56 ± 0.16 | 99.84 ± 0.08 |
| FIS ucgen mamdani centroid (esik 0.5) | 96.96 ± 0.34 | 0.00 ± 0.00 | 98.46 ± 0.17 | 99.52 ± 0.05 |
| FIS ucgen mamdani centroid + ayarli esik | 99.36 ± 0.12 | 0.04 ± 0.05 | 99.67 ± 0.06 | 99.52 ± 0.05 |
| FIS gauss mamdani centroid (esik 0.5) | 96.56 ± 0.34 | 0.00 ± 0.00 | 98.25 ± 0.17 | 99.74 ± 0.07 |
| FIS gauss mamdani centroid + ayarli esik | 99.39 ± 0.13 | 0.11 ± 0.18 | 99.68 ± 0.06 | 99.74 ± 0.07 |
| FIS ucgen mamdani bisector (esik 0.5) | 97.29 ± 0.29 | 0.00 ± 0.00 | 98.63 ± 0.15 | 99.52 ± 0.05 |
| FIS ucgen mamdani bisector + ayarli esik | 99.31 ± 0.13 | 0.02 ± 0.03 | 99.65 ± 0.06 | 99.52 ± 0.05 |
| FIS ucgen mamdani mom (esik 0.5) | 98.80 ± 0.09 | 0.00 ± 0.00 | 99.40 ± 0.04 | 99.44 ± 0.08 |
| FIS ucgen mamdani mom + ayarli esik | 98.94 ± 0.18 | 0.08 ± 0.13 | 99.46 ± 0.08 | 99.44 ± 0.08 |
| FIS ucgen sugeno (esik 0.5) | 97.20 ± 0.31 | 0.00 ± 0.00 | 98.58 ± 0.16 | 99.72 ± 0.02 |
| FIS ucgen sugeno + ayarli esik | 99.36 ± 0.12 | 0.05 ± 0.06 | 99.67 ± 0.06 | 99.72 ± 0.02 |
| FIS gauss sugeno (esik 0.5) | 96.67 ± 0.35 | 0.00 ± 0.00 | 98.31 ± 0.18 | 99.99 ± 0.00 |
| FIS gauss sugeno + ayarli esik | 99.58 ± 0.24 | 0.78 ± 0.85 | 99.69 ± 0.06 | 99.99 ± 0.00 |

| Karsilastirma (vs keskin) | Metrik | Ort. fark | p |
|---|---|---|---|
| FIS ucgen mamdani centroid (esik 0.5) | DR | -1.54 | 0.0020 |
| FIS ucgen mamdani centroid (esik 0.5) | FAR | -0.01 | 0.1250 |
| FIS ucgen mamdani centroid (esik 0.5) | F1 | -0.78 | 0.0020 |
| FIS ucgen mamdani centroid + ayarli esik | DR | +0.86 | 0.0020 |
| FIS ucgen mamdani centroid + ayarli esik | FAR | +0.03 | 0.2188 |
| FIS ucgen mamdani centroid + ayarli esik | F1 | +0.43 | 0.0020 |
| FIS gauss sugeno + ayarli esik | DR | +1.09 | 0.0020 |
| FIS gauss sugeno + ayarli esik | FAR | +0.77 | 0.0039 |
| FIS gauss sugeno + ayarli esik | F1 | +0.45 | 0.0020 |
| Kapsama skoru + ayarli esik | DR | +0.70 | 0.0020 |
| Kapsama skoru + ayarli esik | FAR | +0.22 | 0.0020 |
| Kapsama skoru + ayarli esik | F1 | +0.32 | 0.0020 |

| Karsilastirma (vs kapsama skoru + ayarli esik) | Metrik | Ort. fark | p |
|---|---|---|---|
| FIS ucgen mamdani centroid + ayarli esik | F1 | +0.11 | 0.0273 |
| FIS ucgen mamdani centroid + ayarli esik | FAR | -0.19 | 0.0039 |
| FIS ucgen mamdani centroid + ayarli esik | AUC | -0.32 | 0.0020 |
| FIS gauss sugeno + ayarli esik | F1 | +0.13 | 0.0137 |
| FIS gauss sugeno + ayarli esik | FAR | +0.55 | 0.1309 |
| FIS gauss sugeno + ayarli esik | AUC | +0.15 | 0.0020 |


## E7 gurultu (validasyonda secilen esik)

### bot-iot-clean

|   sigma |   ('DR', 'Bulanik') |   ('DR', 'Keskin') |   ('FAR', 'Bulanik') |   ('FAR', 'Keskin') |   ('F1', 'Bulanik') |   ('F1', 'Keskin') |
|--------:|--------------------:|-------------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|    0    |               98.56 |              98.6  |                 5.26 |                1.71 |               99.05 |              99.22 |
|    0.02 |               98.54 |              98.61 |                 4.93 |                1.8  |               99.05 |              99.22 |
|    0.05 |               98.6  |              98.6  |                 5.4  |                3.89 |               99.06 |              99.13 |
|    0.08 |               98.67 |              98.41 |                 9.1  |                7.01 |               98.94 |              98.9  |
|    0.12 |               98.98 |              98.1  |                34.5  |               12.65 |               98.04 |              98.5  |

### skin

|   sigma |   ('DR', 'Bulanik') |   ('DR', 'Keskin') |   ('FAR', 'Bulanik') |   ('FAR', 'Keskin') |   ('F1', 'Bulanik') |   ('F1', 'Keskin') |
|--------:|--------------------:|-------------------:|---------------------:|--------------------:|--------------------:|-------------------:|
|    0    |               99.36 |              98.49 |                 0.04 |                0.01 |               99.67 |              99.24 |
|    0.02 |               99.32 |              98.37 |                 0.2  |                0.12 |               99.63 |              99.16 |
|    0.05 |               99.22 |              97.86 |                14.17 |                6.54 |               97.8  |              98.07 |
|    0.08 |               98.91 |              96.86 |                38.37 |               20.69 |               94.69 |              95.78 |
|    0.12 |               98.07 |              94.63 |                60.71 |               38.69 |               91.69 |              92.44 |

