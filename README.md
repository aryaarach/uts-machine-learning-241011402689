# UTS Machine Learning – Linear Regression & Logistic Regression

| | |
|---|---|
| **Nama** | Arya Rachman Indrawan |
| **NIM** | 241011402689 |
| **Kelas** | Reg B |
| **Mata Kuliah** | Machine Learning |
| **random_state** | 89 (2 digit terakhir NIM) |

Model dan metrik evaluasi diimplementasikan sendiri dengan NumPy (`train_test_split`, `LinearRegression`, `LinearRegressionGD`, `LogisticRegression`, metrik regresi dan klasifikasi). `scikit-learn` hanya dipakai pada soal bonus untuk perbandingan.

## Isi Repositori

```
uts-machine-learning-241011402689/
├── README.md
├── tugas_UTS.ipynb                # tugas uts machine learning, lengkap dengan output
├── Ads_Data.csv                   # 40 data (BiayaIklan, Penjualan)
└── Email_Data.csv                 # 60 data (JumlahLink, KataPromo, Spam)
```

## Ringkasan Hasil

### Bagian A – Linear Regression (`Ads_Data.csv`)

Split 80:20 (32 training, 8 testing).

**Persamaan (normal equation):** ŷ = 27.5570 + 9.1762 · BiayaIklan
Setiap kenaikan biaya iklan 1 juta Rp, penjualan diprediksi naik sekitar 9,18 juta Rp.

| Metrik | Training | Testing |
|---|---|---|
| MAE | 4.4018 | 5.0772 |
| MSE | 35.7863 | 37.6794 |
| RMSE | 5.9822 | 6.1384 |
| R² | 0.9456 | 0.9246 |

**Gradient descent (5000 iterasi):**

| Learning rate | Intercept | Koefisien | Hasil |
|---|---|---|---|
| 0.001 | 18.4763 | 10.5492 | Terlalu lambat, belum konvergen (loss 53.26) |
| 0.02 | 27.5570 | 9.1762 | Konvergen, sama dengan normal equation (loss 35.79) |
| 0.1 | NaN | NaN | Divergen, overflow sekitar iterasi ke-370 |

**Prediksi penjualan:** biaya iklan 4,5 juta → 68,85 juta; 7 juta → 91,79 juta; 12 juta → 137,67 juta (ekstrapolasi, kurang dapat dipercaya karena di luar rentang data 0,8–9,9 juta).

### Bagian B – Logistic Regression (`Email_Data.csv`)

Split 75:25 (45 training, 15 testing). Gradient descent, learning rate 0.1, 10.000 iterasi.

**Persamaan:** P(Spam) = σ(−12.5226 + 1.0106 · JumlahLink + 1.0251 · KataPromo)
Log loss turun dari 0.6931 menjadi 0.1657.

**Evaluasi data testing (threshold 0.5):**

| Metrik | Nilai |
|---|---|
| Accuracy | 73.33% |
| Precision | 100% |
| Recall | 63.64% |
| F1-Score | 77.78% |
| Confusion matrix [[TN, FP], [FN, TP]] | [[4, 0], [4, 7]] |

**Decision boundary:** x₂ = 12.2164 − 0.9859 · x₁. Ada 4 data testing yang salah klasifikasi, seluruhnya *false negative* (spam yang lolos).

**Eksperimen threshold:**

| Threshold | [[TN, FP], [FN, TP]] | Precision | Recall |
|---|---|---|---|
| 0.3 | [[3, 1], [2, 9]] | 0.9000 | 0.8182 |
| 0.5 | [[4, 0], [4, 7]] | 1.0000 | 0.6364 |
| 0.7 | [[4, 0], [6, 5]] | 1.0000 | 0.4545 |

Threshold yang dipilih adalah **0.5**: *false positive* (email sah masuk folder Spam) lebih merugikan pengguna, dan pada 0.5 FP sudah 0 sedangkan menaikkannya ke 0.7 hanya menurunkan recall.

**Prediksi email baru (threshold 0.5):**

| Email | JumlahLink | KataPromo | P(Spam) | Prediksi |
|---|---|---|---|---|
| E1 | 1 | 2 | 0.0001 | Bukan Spam |
| E2 | 5 | 6 | 0.2109 | Bukan Spam |
| E3 | 9 | 10 | 0.9989 | Spam |
| E4 | 2 | 11 | 0.6844 | Spam |

### Bagian C – Pemahaman Konsep

Lima pertanyaan konsep dijawab di notebook (Logistic Regression sebagai klasifikasi dan peran sigmoid, normal equation vs gradient descent, data testing dan overfitting, MSE vs log loss, pengaruh `random_state`).

### Bonus – Perbandingan dengan scikit-learn

- **Linear Regression:** identik dengan `sklearn.linear_model.LinearRegression` hingga 6 desimal.
- **Logistic Regression:**

  | | Intercept | m1 | m2 |
  |---|---|---|---|
  | Model sendiri (GD) | -12.5226 | 1.0106 | 1.0251 |
  | sklearn tanpa regularisasi (`C=inf`) | -15.1499 | 1.2351 | 1.2364 |
  | sklearn default (L2, `C=1.0`) | -11.9559 | 0.9538 | 0.9776 |

  Default scikit-learn memakai regularisasi L2 sehingga koefisiennya menyusut. Model sendiri belum mencapai optimum setelah 10.000 iterasi sehingga berbeda dari versi tanpa regularisasi. Metrik testing ketiganya identik (accuracy 0.7333, precision 1.0, recall 0.6364, F1 0.7778).

## Cara Menjalankan

Library yang dibutuhkan: `numpy`, `pandas`, `matplotlib` (dan `scikit-learn` hanya untuk bonus). Letakkan kedua file CSV di folder yang sama dengan notebook, lalu buka dan jalankan `tugas_UTS.ipynb` dengan *Run All*.
