# Veri Setleri

Ham veri dosyaları boyut ve lisans nedeniyle repoya yüklenmez (`.gitignore`). Aşağıdaki dosyaları indirip `data/raw/` klasörüne koyun.

| Veri seti | Kullanıldığı deney | Kaynak | Beklenen dosya |
|---|---|---|---|
| Iris | E1 (makale Tablo 7) | scikit-learn içinde gelir, indirme gerekmez | — |
| Skin Segmentation | E2 (makale Tablo 8) | UCI Machine Learning Repository, "Skin Segmentation" (ID 229) | `data/raw/Skin_NonSkin.txt` |
| BoT-IoT | E3 (makale Tablo 5) | UNSW Canberra, BoT-IoT dataset (%5 alt küme veya "10-best features" CSV'leri) | `data/raw/botiot/*.csv` |

## Ön işleme (makale Bölüm 4 ile aynı)

1. Tekrar eden ve eksik (null/inf) satırlar silinir.
2. Kategorik özellikler `LabelEncoder` ile sayısallaştırılır.
3. Tüm özellikler min–max ile [0, 1] aralığına ölçeklenir.

Kod: `src/data.py`.
