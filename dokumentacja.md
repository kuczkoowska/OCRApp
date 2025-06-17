# OCRAppModels

**OCRAppModels** to aplikacja do rozpoznawania tekstu na obrazach (OCR) z wykorzystaniem głębokich sieci neuronowych (CRNN) oraz frameworka TensorFlow. Zawiera narzędzia do trenowania, ewaluacji i udostępniania modeli OCR przez API.

---

## Spis treści

- [Struktura projektu](#struktura-projektu)
- [Wymagania](#wymagania)
- [Baza danych](#baza-danych)
- [Szybki start](#szybki-start)
- [Trenowanie modelu](#trenowanie-modelu)
- [API (FastAPI)](#api-fastapi)

---

## Struktura projektu

```
OCRAppModels/
├── src/
│   ├── api/
│   │   └── app.py           # Serwer FastAPI do predykcji OCR
│   ├── models/
│   │   ├── CRNN.py          # Definicja modelu CRNN
│   │   └── vit.py           # Definicja modelu ViT (nieużywany)
│   ├── results/             # Wyniki i zapisane modele
│   └── train/
│       ├── mjs.py           # Loader i preprocessing danych MJSynth
│       ├── training.py      # Skrypt do trenowania modeli
│       └── evaluate.py      # Ewaluacja modeli OCR
├── mjsynth/                 # Dane MJSynth (obrazy, adnotacje)
├── results/                 # Wyniki porównań modeli
└── models/                  # Zapisane modele i logi TensorBoard
```

---

## Wymagania

- Python 3.11+
- TensorFlow 2.x
- numpy, pillow, editdistance, matplotlib, fastapi, uvicorn, pydantic

Instalacja zależności:
```sh
pip install tensorflow numpy pillow editdistance matplotlib fastapi uvicorn pydantic
```

---

## Baza danych

W projekcie wykorzystywany jest zbiór danych:  
**Reading Text in the Wild with Convolutional Neural Networks**  
[Link do pobrania (Academic Torrents)](https://academictorrents.com/details/3d0b4f09080703d2a9c6be50715b46389fdb3af1)

---

## Szybki start

**Trenowanie modelu:**
```sh
python src/train/training.py
```

**Ewaluacja modelu:**
```sh
python src/train/evaluate.py
```

**Uruchomienie API:**
```sh
uvicorn src.api.app:app --reload
```

---

## Trenowanie modelu

Trenowanie odbywa się przez uruchomienie `training.py`.  
Możesz skonfigurować parametry (`batch_size`, liczba epok, limity danych) w sekcji `config` tego pliku.  
Należy także zmienić ścieżkę do swojego folderu `mjsynth`.

Wyniki treningu (modele, logi, wykresy) zapisywane są w katalogu `models/`.

---

## API (FastAPI)

Serwer API (`src/api/app.py`) udostępnia endpoint `/predict` do rozpoznawania tekstu na przesłanym obrazie.

Przykład użycia (curl):
```sh
curl -X POST "http://localhost:8000/predict" -F "file=@ścieżka/do/obrazu.png"
```

**Link do frontendu:**  
*https://github.com/kuczkoowska/inteligencja*

---

