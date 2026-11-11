# Veri setlerini GitHub'daki kopyalardan indirir (toplam ~68 MB).
# calistirma: python -m src.download_data
# kaynaklar data/README.md'de
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SKIN_URL = ("https://raw.githubusercontent.com/amirshnll/skin-segmentation-classification/"
            "master/Skin_NonSkin.txt")
BOTIOT_URL = ("https://raw.githubusercontent.com/cfoh/IoT-Network-Intrusion-Detection-with-"
              "Ensemble-Learners/main/UNSW_2018_IoT_Botnet_Full5pc_{}.zip")

# beklenen dosya boyutlari, yarim inen dosyayi yakalamak icin
SIZES = {"Skin_NonSkin.txt": 3155761,
         "UNSW_2018_IoT_Botnet_Full5pc_1.zip": 17710175,
         "UNSW_2018_IoT_Botnet_Full5pc_2.zip": 16738990,
         "UNSW_2018_IoT_Botnet_Full5pc_3.zip": 18279640,
         "UNSW_2018_IoT_Botnet_Full5pc_4.zip": 12135612}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size == SIZES[dest.name]:
        print("zaten var:", dest.name)
        return
    print("indiriliyor:", dest.name, f"({SIZES[dest.name] / 1e6:.1f} MB)")
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size != SIZES[dest.name]:
        raise RuntimeError(f"{dest.name} eksik indi, tekrar calistirin")


def main():
    (RAW / "botiot").mkdir(parents=True, exist_ok=True)
    fetch(SKIN_URL, RAW / "Skin_NonSkin.txt")
    for i in range(1, 5):
        z = RAW / f"UNSW_2018_IoT_Botnet_Full5pc_{i}.zip"
        fetch(BOTIOT_URL.format(i), z)
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if name.endswith(".csv") and "/" not in name and ".." not in name:
                    f.extract(name, RAW / "botiot")
    print("tamam. BoT-IoT alt kumesi ilk kullanimda otomatik olusturulacak.")


if __name__ == "__main__":
    main()
