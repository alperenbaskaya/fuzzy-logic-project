# Veri Setleri

Ham veri dosyaları boyut ve lisans nedeniyle repoya yüklenmez (`.gitignore`). Hepsini indirip `data/raw/` klasörüne yerleştirmek için:

```bash
python -m src.download_data
```

Betik dosya boyutlarını kontrol eder; yarım kalan bir indirmede tekrar çalıştırmak yeterlidir.

| Veri seti | Kullanıldığı deney | Kaynak | Beklenen dosya |
|---|---|---|---|
| Iris | E1 (makale Tablo 7) | scikit-learn içinde gelir, indirme gerekmez | — |
| Skin Segmentation | E2, E4 (makale Tablo 8) | UCI Machine Learning Repository, "Skin Segmentation" (ID 229). Kullanılan kopya: [amirshnll/skin-segmentation-classification](https://github.com/amirshnll/skin-segmentation-classification) (245.057 satır, UCI dosyasıyla aynı) | `data/raw/Skin_NonSkin.txt` |
| BoT-IoT | E3, E4 (makale Tablo 5) | UNSW Canberra, BoT-IoT dataset, resmî %5 alt küme. Kullanılan kopya: [cfoh/IoT-Network-Intrusion-Detection-with-Ensemble-Learners](https://github.com/cfoh/IoT-Network-Intrusion-Detection-with-Ensemble-Learners) (`UNSW_2018_IoT_Botnet_Full5pc_1-4.zip`, toplam 3.668.522 akış, 477 normal) | `data/raw/botiot/*.csv` |

İlk çalıştırmada `src/data.py` içindeki `build_botiot_subset()` 477 normalin tamamı ve kategoriye orantılı 5.050 saldırıdan oluşan `data/raw/botiot_subset.csv` dosyasını bir kez oluşturur (ASSUMPTIONS A12).

## Ön işleme (makale Bölüm 4 ile aynı)

1. Tekrar eden ve eksik (null/inf) satırlar silinir.
2. Kategorik özellikler `LabelEncoder` ile sayısallaştırılır.
3. Tüm özellikler min–max ile [0, 1] aralığına ölçeklenir.

Kod: `src/data.py`.
