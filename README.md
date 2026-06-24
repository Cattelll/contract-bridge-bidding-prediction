# Prediksi Kontrak Terbaik pada Permainan Bridge Menggunakan Algoritma Neural Network Berbasis Data BBO

Proyek skripsi S1 Ilmu Komputer yang bertujuan memprediksi kontrak terbaik dalam permainan bridge menggunakan algoritma **two-stage neural network** (MLP dan LSTM) dengan data dari Bridge Base Online (BBO).

## Deskripsi Proyek

Penelitian ini mengimplementasikan pipeline machine learning untuk memprediksi kontrak bridge terbaik dengan dua tahap:

1. **Stage 1**: Prediksi suit kontrak (Club, Diamond, Heart, Spade, NoTrump)
2. **Stage 2**: Prediksi kategori kontrak (Partial Game, Game, Slam, Grand Slam)

Data mentah berupa file `.lin` dari BBO yang diproses melalui parsing, feature extraction, dan labeling menggunakan Double Dummy Solver (DDS).

## Struktur Direktori

```
SkripsiBBO/
├── Claude/                 # Konteks proyek
├── data/
│   ├── parsed/            # Hasil parsing file .lin
│   ├── processed/         # Dataset dan fitur hasil processing
│   └── raw/               # File BBO .lin mentah (jika ada)
├── docs/                  # Dokumentasi tambahan
├── notebooks/             # Jupyter notebooks untuk eksperimen
│   ├── 01_parsing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_features.ipynb
│   ├── 04_labeling.ipynb
│   ├── 05_dataset.ipynb
│   ├── 06_training.ipynb
│   ├── 07_evaluasi.ipynb
│   └── 08_analisis.ipynb
├── results/               # Semua hasil (model, metrics, figures)
│   ├── figures/           # Visualisasi hasil
│   └── metrics/           # Hasil evaluasi dan model tersimpan
│       ├── mlp/           # Trained MLP model files
│       └── lstm/          # Trained LSTM model files
├── src/                   # Source code utama
│   ├── parser.py          # Parser file .lin
│   ├── features.py        # Feature extraction
│   ├── labeling.py        # DDS labeling
│   ├── model.py           # Training model (MLP & LSTM)
│   └── evaluate.py        # Evaluasi model
├── requirements.txt       # Dependencies
└── README.md
```

## Instalasi

1. Clone repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

- **pandas**: Manipulasi data
- **scikit-learn**: Machine learning utilities
- **numpy**: Operasi numerik
- **matplotlib & seaborn**: Visualisasi
- **tensorflow & keras**: Deep learning (MLP & LSTM)
- **endplay**: Double Dummy Solver
- **ipykernel & notebook**: Jupyter Notebook

## Cara Menggunakan

### 1. Parsing Data

Jalankan `notebooks/01_parsing.ipynb` untuk memparse file `.lin` menjadi CSV.

### 2. Exploratory Data Analysis (EDA)

Jalankan `notebooks/02_eda.ipynb` untuk eksplorasi data.

### 3. Feature Extraction

Jalankan `notebooks/03_features.ipynb` untuk mengekstrak fitur dari dataset.

### 4. Labeling dengan DDS

Jalankan `notebooks/04_labeling.ipynb` untuk membuat label menggunakan Double Dummy Solver.

### 5. Dataset Preparation

Jalankan `notebooks/05_dataset.ipynb` untuk mempersiapkan dataset final.

### 6. Training Model

Jalankan `notebooks/06_training.ipynb` untuk training **kedua model (MLP dan LSTM)** secara otomatis!

Notebook ini:

- Melatih 2-stage MLP untuk tabular features
- Melatih 2-stage LSTM (mereshape fitur ke sequence)
- Menyimpan model di `results/metrics/mlp` dan `results/metrics/lstm` (format Keras .keras)
- Menampilkan prediksi sanity check

### 7. Evaluasi

Jalankan `notebooks/07_evaluasi.ipynb` untuk mengevaluasi **kedua model (MLP & LSTM)**!

Notebook ini:

- Memuat model dari `results/metrics/mlp` dan `results/metrics/lstm`
- Evaluasi menggunakan 7 indikator evaluasi
- Menampilkan confusion matrix dan feature importance
- Menyimpan semua hasil di `results/`
- Bandingkan performa kedua model dan kinerjanya!

### 8. Analisis

Jalankan `notebooks/08_analisis.ipynb` untuk analisis hasil.

## Model Tersedia

| Model    | Arsitektur                                                                                             | Kelebihan                                | Format File    | Direktori Penyimpanan  |
| -------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------- | -------------- | ---------------------- |
| **MLP**  | Input (98) → Dense(256) + Dropout(0.3) → Dense(128) + Dropout(0.3) → Dense(64) + Dropout(0.3) → Output | Training cepat, cocok untuk data tabular | Keras (.keras) | `results/metrics/mlp`  |
| **LSTM** | Input (10×9) → LSTM(128) + Dropout(0.3) → LSTM(64) + Dropout(0.3) → Dense(64) + Dropout(0.2) → Output  | Capture temporal dependencies            | Keras (.keras) | `results/metrics/lstm` |

## Fitur yang Diekstrak

- **Hand Features**: HCP per suit, jumlah kartu per suit, total HCP
- **Distribusi**: Balanced/unbalanced label, combined HCP NS
- **Stopper**: Stopper per suit (0-3 encoding)
- **Situasi**: Vulnerability, dealer
- **Bidding History**: Encoded bidding sequence

## Evaluasi

Model dievaluasi menggunakan:

- Akurasi per stage
- Confusion matrix
- 7 indikator evaluasi (MS, SCA, SCU, SSE, O, SS, SC)
- IMP score (bandingkan dengan kontrak BBO)

## Referensi

- Lin et al. (2023) - _Two-stage Random Forest for Non-competitive Bidding_
- Chen & Yang (2025) - _Neural Network for Competitive Bidding_

## Lisensi

Proyek ini untuk keperluan skripsi S1 Ilmu Komputer.

## Catatan Penting

- Semua model, figure, dan hasil evaluasi disimpan di direktori **`results/`** (root project, bukan di `notebooks/results/`)
- File model, metrics, dan figures diignore oleh git (lihat `.gitignore`)
