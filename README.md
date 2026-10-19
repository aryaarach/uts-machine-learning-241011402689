# Ujian Tengah Semester (UTS) Machine Learning: Linear & Logistic Regression from Scratch

Dokumentasi komprehensif implementasi algoritma **Linear Regression** dan **Logistic Regression** yang dibangun murni dari dasar (*from scratch*) menggunakan bahasa pemrograman Python dan pustaka aljabar linear **NumPy**. Seluruh fungsi pemodelan, optimasi, dan metrik evaluasi dibangun tanpa pustaka *machine learning black-box* seperti Scikit-Learn (Scikit-Learn hanya difungsikan pada bagian pengujian validasi komparatif).

---

## 📌 Identitas Mahasiswa & Parameter Pengacakan

| Parameter | Keterangan |
|---|---|
| **Nama Lengkap** | **Arya Rachman Indrawan** |
| **Nomor Induk Mahasiswa (NIM)** | `241011402689` |
| **Kelas** | Reguler B (Reg B) |
| **Mata Kuliah** | Machine Learning |
| **Dua Digit Terakhir NIM** | `89` |
| **Nilai Seed `random_state`** | `89` *(digunakan secara konsisten untuk partisi data latih dan uji)* |

---

## 💡 Ringkasan Hasil Eksekutif

| Aspek Evaluasi | Bagian A: Linear Regression | Bagian B: Logistic Regression |
|---|---|---|
| **Nama Dataset** | `Ads_Data.csv` (Biaya Iklan vs Penjualan) | `Email_Data.csv` (Jumlah Link & Kata Promo vs Spam) |
| **Jumlah Sampel** | 40 baris observasi | 60 baris observasi |
| **Rasio Partisi (Train : Test)** | 80% : 20% (32 data latih : 8 data uji) | 75% : 25% (45 data latih : 15 data uji) |
| **Metode Optimasi** | *Normal Equation* & *Gradient Descent* | *Binary Cross-Entropy Gradient Descent* |
| **Persamaan Matematika** | $\hat{y} = 27.5570 + 9.1762 \cdot \text{BiayaIklan}$ | $P(\text{Spam}) = \sigma(-12.5226 + 1.0106 x_1 + 1.0251 x_2)$ |
| **Metrik Kinerja Utama (Testing)** | **$R^2 = 0.9246$ (92.46%)**, $\text{RMSE} = 6.14\text{ juta Rp}$ | **$\text{Accuracy} = 73.33\%$**, **$\text{Precision} = 100.0\%$** |
| **Validasi Scikit-Learn** | **Identik 100% hingga 6 digit desimal** | **Metrik klasifikasi testing 100% konsisten** |

---

## 📂 Struktur Repositori

```text
uts-machine-learning-241011402689/
├── README.md            # Dokumentasi akademik lengkap, pembahasan soal, & fungsi
├── tugas_UTS.ipynb      # Jupyter Notebook pengerjaan (sudah di-Run All lengkap output & grafik)
├── Ads_Data.csv         # Dataset Biaya Iklan vs Penjualan (40 data)
└── Email_Data.csv       # Dataset Jumlah Link & Kata Promo vs Spam (60 data)
```

---

## 🛠️ Arsitektur & Implementasi Fungsi Mandiri (*From Scratch*)

Seluruh modul komputasi inti diimplementasikan menggunakan operasi vektor dan matriks NumPy:

### 1. Fungsi Partisi Data (`train_test_split`)
Fungsi membagi dataset menjadi subset pelatihan (*training*) dan pengujian (*testing*) menggunakan generator bilangan acak berbasis *seed* deterministik (`random_state`).

```python
def train_test_split(X, y, test_size=0.25, random_state=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    n_total = len(X)
    n_test = int(round(n_total * test_size))

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_total)
    else:
        indices = np.random.permutation(n_total)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

---

### 2. Kelas `LinearRegression` (Solusi Analitik: *Normal Equation*)
Menyelesaikan estimasi parameter kuadrat terkecil (*ordinary least squares*) secara analitik dalam satu langkah aljabar matriks tertutup (*closed-form solution*):

$$
\boldsymbol{\beta} = (\mathbf{X}_b^T \mathbf{X}_b)^{-1} \mathbf{X}_b^T \mathbf{y}
$$

```python
class LinearRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficients_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Menambahkan kolom bias (vektor bernilai 1)
        X_b = np.c_[np.ones(len(X)), X]
        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = beta[0]
        self.coefficients_ = beta[1:]
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coefficients_
```

---

### 3. Fungsi Metrik Evaluasi Regresi (`MAE`, `MSE`, `RMSE`, `R²`)
Mengukur performa aproksimasi nilai numerik kontinu:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \qquad
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \qquad
\text{RMSE} = \sqrt{\text{MSE}}
$$

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

```python
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
```

---

### 4. Kelas `LinearRegressionGD` (Solusi Iteratif: *Gradient Descent*)
Memperbarui vektor bobot $\boldsymbol{\beta}$ secara iteratif dengan menuruni kurva gradien fungsi loss MSE:

$$
\nabla_{\boldsymbol{\beta}} L = \frac{2}{n} \mathbf{X}_b^T (\mathbf{X}_b \boldsymbol{\beta} - \mathbf{y}) \quad \longrightarrow \quad \boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \nabla_{\boldsymbol{\beta}} L
$$

```python
class LinearRegressionGD:
    def __init__(self, learning_rate=0.02, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = 0.0
        self.coefficients_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_b = np.c_[np.ones(len(X)), X]
        beta = np.zeros(X_b.shape[1])

        for _ in range(self.n_iterations):
            y_hat = X_b @ beta
            gradient = X_b.T @ (y_hat - y) / len(y)
            beta = beta - self.learning_rate * gradient

            loss = np.mean((y - X_b @ beta) ** 2)
            self.loss_history_.append(loss)

        self.intercept_ = beta[0]
        self.coefficients_ = beta[1:]
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coefficients_
```

---

### 5. Kelas `LogisticRegression` (Klasifikasi Sigmoid & *Log Loss*)
Memetakan kombinasi linear ke probabilitas biner melalui fungsi sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, dengan fungsi objektif *Cross-Entropy* (*Log Loss*):

$$
L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]
$$

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = 0.0
        self.coefficients_ = None
        self.loss_history_ = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_b = np.c_[np.ones(len(X)), X]
        beta = np.zeros(X_b.shape[1])

        for _ in range(self.n_iterations):
            p = self._sigmoid(X_b @ beta)
            gradient = X_b.T @ (p - y) / len(y)
            beta = beta - self.learning_rate * gradient

            p = np.clip(p, 1e-15, 1 - 1e-15)  # Mencegah log(0)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_history_.append(loss)

        self.intercept_ = beta[0]
        self.coefficients_ = beta[1:]
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._sigmoid(self.intercept_ + X @ self.coefficients_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

### 6. Fungsi Metrik Evaluasi Klasifikasi
Mengevaluasi tabel kontingensi dan trade-off presisi vs sensitivitas:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \qquad
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN} \qquad
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

```python
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def accuracy_score(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
```

---

## 📊 Bagian A: Pembahasan Lengkap Soal Linear Regression

Kasus: Pemodelan pengaruh alokasi anggaran **Biaya Iklan** terhadap volume **Penjualan** bulanan sebuah toko online berdasarkan berkas `Ads_Data.csv`.

---

### 🔹 Soal A1 – Pemuatan dan Visualisasi Data
* **Instruksi Soal:** Baca berkas `Ads_Data.csv` menggunakan pandas, tampilkan 5 baris pertama, hitung total baris, bentuk matriks fitur $X$ berdimensi 2D `(n, 1)` dan vektor target $y$, lalu gambarkan scatter plot hubungan kedua variabel.
* **Hasil Eksekusi:**
  * Total data observasi: **40 baris data**.
  * Struktur matriks: `X.shape = (40, 1)`, `y.shape = (40,)`.
  * Rentang nilai variabel independen (`BiayaIklan`): **0.8 s.d. 9.9 juta Rp**.
  * Rentang nilai variabel dependen (`Penjualan`): **31.5 s.d. 120.4 juta Rp**.
* **Analisis Visual:** Scatter plot memperlihatkan pola korelasi positif yang sangat linear dan kuat. Peningkatan pada sumbu horizontal biaya iklan secara konsisten diikuti oleh kenaikan nilai penjualan pada sumbu vertikal, membuktikan kesesuaian penerapan model regresi linear parametrik.

---

### 🔹 Soal A2 – Partisi Data (*Train-Test Split*)
* **Instruksi Soal:** Bagi dataset menjadi data pelatihan (*training*) dan pengujian (*testing*) dengan proporsi `test_size=0.2` dan generator bilangan acak `random_state=89` (2 digit terakhir NIM `241011402689`).
* **Hasil Eksekusi Dimensi Array:**
  * `X_train.shape = (32, 1)` | `y_train.shape = (32,)` $\rightarrow$ 80% dari dataset
  * `X_test.shape  = (8, 1)`  | `y_test.shape  = (8,)`  $\rightarrow$ 20% dari dataset
* **Tujuan Metodologis:** Pembagian subset uji yang terisolasi bertujuan untuk menguji kemampuan generalisasi model terhadap data baru (*unseen data*) yang belum pernah dipelajari pada fase estimasi parameter.

---

### 🔹 Soal A3 – Pelatihan dengan *Normal Equation* & Interpretasi
* **Instruksi Soal:** Latih model regresi linear menggunakan metode *Normal Equation* pada data training. Tampilkan nilai intercept ($b$) dan slope ($m$), rumuskan persamaan regresinya, dan interpretasikan maknanya dalam domain bisnis.
* **Hasil Estimasi Parameter:**
  * **Intercept ($b$):** `27.5570`
  * **Slope / Koefisien ($m$):** `9.1762`
* **Persamaan Garis Regresi Terbentuk:**

  $$
  \hat{y} = 27.5570 + 9.1762 \cdot \text{BiayaIklan}
  $$
* **Interpretasi Akademis & Bisnis:**
  1. **Intercept ($b = 27.5570$ juta Rp):** Merupakan konstanta estimasi penjualan dasar bulanan apabila perusahaan sama sekali tidak mengeluarkan biaya promosi iklan ($X = 0$). Karena titik nol berada sedikit di luar rentang data latih riil (0.8–9.9 juta Rp), angka ini diartikan sebagai acuan dasar omzet alamiah toko (*organic baseline sales*).
  2. **Slope ($m = 9.1762$):** Koefisien regresi bernilai positif, menandakan korelasi searah yang kuat. Secara kuantitatif, **setiap tambahan alokasi biaya iklan sebesar 1 juta Rp diproyeksikan akan meningkatkan omzet penjualan bulanan rata-rata sebesar 9,18 juta Rp**. Rasio pengembalian ini mengindikasikan efektivitas pemasaran yang sangat tinggi (*Return on Ad Spend* / ROAS positif).

---

### 🔹 Soal A4 – Evaluasi Model (Data Training vs Data Testing)
* **Instruksi Soal:** Implementasikan fungsi MAE, MSE, RMSE, dan $R^2$ dari nol menggunakan NumPy. Hitung pada data training dan testing, sajikan dalam tabel perbandingan, jelaskan arti nilai RMSE dan $R^2$, serta tentukan apakah model terindikasi *overfitting*.
* **Tabel Perbandingan Kinerja:**

| Metrik Evaluasi | Data Training (32 Sampel) | Data Testing (8 Sampel) | Penjelasan Makna Metrik |
|---|:---:|:---:|---|
| **MAE** *(Mean Absolute Error)* | **4.4018** | **5.0772** | Rata-rata absolut deviasi prediksi terhadap angka penjualan riil adalah ~5,08 juta Rp. |
| **MSE** *(Mean Squared Error)* | **35.7863** | **37.6794** | Kuadrat rata-rata selisih galat (memberikan bobot penalti tinggi terhadap galat ekstrem). |
| **RMSE** *(Root Mean Squared Error)* | **5.9822** | **6.1384** | Standar deviasi residu dalam satuan juta Rp; rata-rata prediksi meleset sekitar 6,14 juta Rp. |
| **$R^2$ Score** *(Koefisien Determinasi)* | **0.9456 (94.56%)** | **0.9246 (92.46%)** | **92,46%** variansi penjualan pada data uji berhasil dijelaskan oleh variabel biaya iklan. |

* **Analisis Kualitas Model:**
  * Model terbukti **sangat baik (*well-fitted*) dan tidak mengalami *overfitting***.
  * Skor $R^2$ data uji melampaui 92%, membuktikan daya prediksi model yang kokoh.
  * Selisih performa antara data latih ($R^2 = 94.56\%$) dan data uji ($R^2 = 92.46\%$) hanya terpaut $\approx 2.1\%$, dengan deviasi RMSE yang relatif identik (5.98 vs 6.14 juta Rp). Hal ini membuktikan bahwa model mampu menggeneralisasi data baru secara konsisten tanpa menghafal derau (*noise*) sampel latihan.

---

### 🔹 Soal A5 – Eksperimen Dinamika *Learning Rate* pada *Gradient Descent*
* **Instruksi Soal:** Latih model `LinearRegressionGD` sebanyak 5.000 iterasi dengan 3 variasi *learning rate* ($\alpha \in \{0.001, 0.02, 0.1\}$). Tampilkan parameter akhir dan loss MSE, visualisasikan kurva loss vs iterasi (skala linier dan log), serta jelaskan fenomena konvergensi tiap skenario.
* **Tabel Perbandingan Konvergensi:**

| Skenario Learning Rate ($\alpha$) | Intercept ($b$) | Koefisien ($m$) | Loss Akhir (MSE) | Perilaku & Karakteristik Konvergensi |
|---|:---:|:---:|:---:|---|
| **Solusi Analitik (*Normal Eq.*)** | **27.5570** | **9.1762** | **35.7863** | *Global optimum* acuan teoretis mutlak. |
| **$\alpha = 0.001$ (Terlalu Kecil)** | 18.4763 | 10.5492 | 53.2580 | **Konvergensi Sangat Lambat (*Under-shooting*):** Penurunan loss mulus tanpa osilasi namun langkah pembaruan terlalu kecil. Pada iterasi ke-5000, model belum sempat menyentuh dasar cekungan lembah loss. |
| **$\alpha = 0.02$ (Optimal / Pas)** | **27.5570** | **9.1762** | **35.7863** | **Konvergen Sempurna:** Loss merosot tajam (dari 683 ke 38 pada iterasi ke-500) lalu stabil mendatar. Hasil parameternya **identik sempurna** dengan *Normal Equation*. |
| **$\alpha = 0.1$ (Terlalu Besar)** | `NaN` | `NaN` | `overflow` | **Divergensi Ekstrem (*Overshooting*):** Langkah pembaruan bobot melompati titik minimum global ke lereng seberang yang jauh lebih curam. Nilai error melambung eksponensial hingga batas representasi floating-point terlampaui (*overflow* pada iterasi $\approx 370$). |

* **Penjelasan Teoretis `RuntimeWarning: overflow`:** Nilai fitur `BiayaIklan` mencapai angka 9.9 sehingga menghasilkan vektor gradien $\mathbf{X}_b^T (\hat{y} - y)$ bernilai cukup besar. Perkalian gradien besar dengan $\alpha = 0.1$ memicu ledakan besaran angka (*numerical explosion*), menghasilkan nilai `inf` yang kemudian memicu peringatan overflow dan berubah menjadi `NaN`.

---

### 🔹 Soal A6 – Visualisasi Data Uji, Prediksi, & Analisis Ekstrapolasi
* **Instruksi Soal:** Buat grafik *scatter* data testing bersama garis regresi model. Lakukan estimasi penjualan untuk 3 skenario biaya iklan: **4.5 juta Rp**, **7.0 juta Rp**, dan **12.0 juta Rp**. Berikan evaluasi reliabilitas prediksi terhadap bahaya ekstrapolasi.
* **Tabel Hasil Prediksi Nilai Penjualan:**

| Skenario Anggaran Iklan | Proyeksi Penjualan ($\hat{y}$) | Klasifikasi Wilayah Prediksi | Tingkat Reliabilitas Estimasi |
|:---:|:---:|:---:|---|
| **4.5 Juta Rp** | **68.85 Juta Rp** | **Interpolasi** (di dalam interval data) | **Sangat Tinggi / Terpercaya** |
| **7.0 Juta Rp** | **91.79 Juta Rp** | **Interpolasi** (di dalam interval data) | **Sangat Tinggi / Terpercaya** |
| **12.0 Juta Rp** | **137.67 Juta Rp** | **Ekstrapolasi** (di luar interval data) | **Rendah / Harus Diwaspadai** |

* **Analisis Teoretis Ekstrapolasi vs Realitas Bisnis:**
  * Rentang data historis hanya berada pada domain $[0.8, 9.9]$ juta Rp. Nilai 12.0 juta Rp berada jauh melampaui batas observasi empiris.
  * Model regresi linear sederhana mengasumsikan pertumbuhan tanpa batas dengan laju tetap (kenaikan konstan 9,18 juta per 1 juta iklan). Padahal dalam domain ekonomi dan pemasaran riil berlaku **Hukum Kejenuhan Pasar (*Law of Diminishing Marginal Returns*)**. Melebihi ambang tertentu, penambahan anggaran iklan akan mengalami penurunan efektivitas karena audiens target sudah jenuh, sehingga proyeksi penjualan 137,67 juta Rp cenderung **terlalu optimis (*overestimated*)**.

---

## 📧 Bagian B: Pembahasan Lengkap Soal Logistic Regression

Kasus: Pembangunan sistem penyaring email sampah (*spam filter*) biner berdasarkan dua indikator kuantitatif: **Jumlah Link** ($x_1$) dan **Kata Promosi** ($x_2$) menggunakan dataset `Email_Data.csv`.

---

### 🔹 Soal B1 – Pemuatan Data & Visualisasi Sebaran Kelas
* **Instruksi Soal:** Baca `Email_Data.csv`, tampilkan 5 data teratas, hitung sebaran label target `Spam`, pisahkan matriks $X$ dan vektor $y$, serta gambarkan scatter plot 2D sebaran data antar kelas.
* **Hasil Eksplorasi Data:**
  * Total data observasi: **60 email**.
  * Distribusi Kelas Target:
    * Kelas `0` (Bukan Spam / Ham): **30 email (50%)**
    * Kelas `1` (Spam): **30 email (50%)**
  * Dataset memiliki komposisi kelas yang **seimbang sempurna (*balanced dataset*)**.
* **Analisis Scatter Plot:** Email spam (merah segitiga) mengelompok secara terdistribusi pada kuadran kanan atas (jumlah link banyak dan kata promo tinggi), sedangkan email bukan spam (biru bulat) terkonsentrasi di kuadran kiri bawah. Terdapat area tumpang tindih (*overlap*) moderat pada area tengah yang memerlukan batas keputusan probabilitas optimal.

---

### 🔹 Soal B2 – Partisi Data, Pelatihan Model, & Kurva *Log Loss*
* **Instruksi Soal:** Bagi dataset dengan rasio `test_size=0.25` dan `random_state=89`. Latih model `LogisticRegression` (GD, $\alpha = 0.1$, 10.000 iterasi). Tampilkan intercept ($b$) dan koefisien ($m_1, m_2$), tuliskan persamaan peluang sigmoid, dan visualisasikan kurva *Log Loss* terhadap iterasi.
* **Hasil Partisi Data:**
  * Data Training: **45 email** (75%)
  * Data Testing: **15 email** (25%: 4 Bukan Spam, 11 Spam)
* **Hasil Parameter Terestimasi:**
  * **Intercept ($b$):** `-12.5226`
  * **Koefisien $m_1$ (`JumlahLink`):** `1.0106`
  * **Koefisien $m_2$ (`KataPromo`):** `1.0251`
* **Formulasi Persamaan Peluang Sigmoid:**

  $$
  z = -12.5226 + 1.0106 \cdot \text{JumlahLink} + 1.0251 \cdot \text{KataPromo}
  $$

  $$
  P(\text{Spam} \mid X) = \sigma(z) = \frac{1}{1 + e^{-(-12.5226 + 1.0106 \cdot \text{JumlahLink} + 1.0251 \cdot \text{KataPromo})}}
  $$
* **Evaluasi Dinamika Loss:**
  * **Initial Log Loss:** `0.6931` (setara $-\ln(0.5)$, yaitu tebakan acak murni pada probabilitas seimbang).
  * **Final Log Loss:** `0.1657` (turun signifikan dan mendatar stabil).
  * Kedua koefisien bertanda positif dan berbobot seimbang ($\approx +1.01$ dan $+1.02$). Setiap penambahan 1 link mengalikan rasio kemungkinan (*odds*) spam sebesar $e^{1.0106} \approx 2.75$ kali, dan setiap penambahan 1 kata promo melipatgandakan *odds* spam sebesar $e^{1.0251} \approx 2.79$ kali.

---

### 🔹 Soal B3 – Evaluasi Kinerja Klasifikasi (Threshold Default 0.5)
* **Instruksi Soal:** Implementasikan fungsi metrik klasifikasi mandiri (`accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`). Evaluasi pada 15 data uji pada ambang batas default 0.5, serta jelaskan makna FP dan FN dalam sistem filter spam beserta tingkat kekritisan risikonya.
* **Confusion Matrix Data Testing (15 Email Uji):**

$$
\text{Confusion Matrix} = \begin{bmatrix} TN = 4 & FP = 0 \\ FN = 4 & TP = 7 \end{bmatrix}
$$

* **Tabel Metrik Evaluasi Testing:**

| Metrik Evaluasi | Nilai Kinerja | Interpretasi Matematis dan Praktis |
|---|:---:|---|
| **Akurasi (*Accuracy*)** | **73.33%** | Sebanyak 11 dari total 15 sampel uji berhasil diklasifikasikan dengan benar ($11/15$). |
| **Presisi (*Precision*)** | **100.0%** | Dari seluruh email yang divonis "Spam" oleh model, **100% terbukti benar-benar spam** ($7/7$). Tidak ada satupun email legal yang salah tangkap. |
| **Sensitivitas (*Recall*)** | **63.64%** | Model sukses menyaring 7 dari total 11 email spam riil ($7/11$). Sebanyak 4 email spam lolos. |
| **F1-Score** | **77.78%** | Rata-rata harmonik antara presisi sempurna (1.0) dan recall moderat (0.64). |

* **Analisis Dampak Kesalahan: False Positive (FP) vs False Negative (FN):**
  * **False Positive (FP):** Email biasa (*Bukan Spam*) yang keliru divonis sebagai *Spam* dan dipindahkan ke folder sampah. Contoh: email penawaran kerja, email resmi universitas/dosen, verifikasi perbankan, atau kode OTP.
  * **False Negative (FN):** Email *Spam* berbahaya/mengganggu yang lolos dari saringan dan masuk ke inbox utama pengguna.
  * **Kesimpulan Kekritisan:** **False Positive jauh lebih berbahaya (*more catastrophic*)**. Dalam arsitektur penyaring spam komersial, hilangnya informasi penting akibat masuk folder spam dapat merugikan reputasi, finansial, dan akademis pengguna. Sementara konsekuensi dari FN hanya mengharuskan pengguna meluangkan sedikit waktu untuk menghapus email sampah secara manual. Oleh karena itu, capaian **$\text{FP} = 0$ ($\text{Presisi} = 100\%$)** pada model ini merupakan karakteristik operasional yang sangat ideal.

---

### 🔹 Soal B4 – Penurunan Matematis *Decision Boundary* & Analisis Error
* **Instruksi Soal:** Turunkan persamaan garis *decision boundary* analitik saat probabilitas $P(\text{Spam}) = 0.5$. Buat visualisasi kontur probabilitas (`contourf`), garis batas keputusan, sebaran data testing, serta tandai sampel yang mengalami salah klasifikasi.
* **Penurunan Matematis Persamaan Garis:**
  Garis batas keputusan tercapai tepat ketika probabilitas bernilai 0.5, yaitu saat argumen logit $z = 0$:

  $$
  \begin{aligned}
  b + m_1 x_1 + m_2 x_2 &= 0 \\
  -12.5226 + 1.0106 \cdot \text{JumlahLink} + 1.0251 \cdot \text{KataPromo} &= 0 \\
  \text{KataPromo} &= -\frac{-12.5226 + 1.0106 \cdot \text{JumlahLink}}{1.0251} \\
  \mathbf{\text{KataPromo}} &= \mathbf{12.2164 - 0.9859 \cdot \text{JumlahLink}}
  \end{aligned}
  $$
* **Identifikasi Sampel Uji yang Salah Diklasifikasikan:**
  Terdapat **4 titik data uji yang salah diprediksi**, dan seluruhnya bertipe **False Negative (FN)**:

| ID Baris Asli | Fitur: JumlahLink ($x_1$) | Fitur: KataPromo ($x_2$) | Nilai Logit ($z$) | Probabilitas $P(\text{Spam})$ | Label Aktual | Label Prediksi |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baris #12** | 6 | 4 | -2.3587 | **0.0864 (8.64%)** | 1 (Spam) | 0 (Bukan Spam) |
| **Baris #29** | 5 | 6 | -1.3189 | **0.2109 (21.09%)** | 1 (Spam) | 0 (Bukan Spam) |
| **Baris #34** | 4 | 7 | -0.3041 | **0.4246 (42.46%)** | 1 (Spam) | 0 (Bukan Spam) |
| **Baris #47** | 7 | 4 | -1.3481 | **0.2062 (20.62%)** | 1 (Spam) | 0 (Bukan Spam) |

* **Penyebab Klasifikasi Keliru:** Keempat email spam tersebut menggunakan teknik penulisan cerdik dengan membatasi jumlah link (hanya 4–7 link) dan meminimalkan kata promosi eksplisit (hanya 4–7 kata), sehingga koordinat fiturnya berada di bawah garis ambang $z=0$ dan tidak terdeteksi oleh model linier sederhana.

---

### 🔹 Soal B5 – Eksperimen Ambang Batas (*Classification Threshold*)
* **Instruksi Soal:** Uji coba variasi *threshold* probabilitas pada nilai **0.3**, **0.5**, dan **0.7**. Sajikan tabel confusion matrix, precision, dan recall. Jelaskan pengaruh perubahan ambang batas dan tetapkan threshold operasional terbaik beserta alasannya.
* **Tabel Perbandingan Kinerja Lintas Threshold:**

| Ambang Batas (*Threshold*) | Confusion Matrix `[[TN, FP], [FN, TP]]` | Presisi (*Precision*) | Sensitivitas (*Recall*) | Analisis Konsekuensi Operasional |
|:---:|:---:|:---:|:---:|---|
| **`0.3` (Agresif / Sensitif)** | `[[3, 1], [2, 9]]` | **90.00%** | **81.82%** | Recall naik tajam (9 spam terjaring), **namun muncul 1 False Positive**. Ada email penting yang terbuang ke folder spam. |
| **`0.5` (Optimal / Terpilih)** | `[[4, 0], [4, 7]]` | **100.0%** | **63.64%** | **Zero False Positive ($\text{FP} = 0$)**. Keamanan inbox legal terjamin mutlak tanpa ada email penting yang salah buang. |
| **`0.7` (Ketat / Konservatif)** | `[[4, 0], [6, 5]]` | **100.0%** | **45.45%** | Presisi tetap 100%, namun Recall jatuh parah. Lebih dari separuh email spam lolos membobol inbox utama. |

* **Justifikasi Pemilihan Threshold:**
  * **Ambang batas 0.5 ditetapkan sebagai opsi terbaik.**
  * Menurunkan ambang ke 0.3 meningkatkan recall namun mengorbankan presisi (mengakibatkan 1 email penting menjadi korban salah tangkap). Menaikkan ambang ke 0.7 membuat model terlalu permisif terhadap spam. Threshold 0.5 menghadirkan ekuilibrium terbaik yang selaras dengan prinsip pertahanan sistem email: **memprioritaskan perlindungan email penting dari salah filter**.

---

### 🔹 Soal B6 – Prediksi Klasifikasi Email Baru
* **Instruksi Soal:** Lakukan inferensi probabilitas dan penentuan label prediksi untuk 4 sampel email baru menggunakan threshold terpilih 0.5.
* **Tabel Inferensi Email Baru:**

| ID Email | JumlahLink ($x_1$) | KataPromo ($x_2$) | Nilai Logit ($z$) | Probabilitas Sigmoid $P(\text{Spam})$ | Keputusan Model | Rasionalisasi Prediksi |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **E1** | 1 | 2 | -9.4618 | **0.0001 (0.01%)** | 🟢 **Bukan Spam** | Nilai fitur sangat rendah. Titik berada jauh di bawah garis keputusan. |
| **E2** | 5 | 6 | -1.3189 | **0.2109 (21.09%)** | 🟢 **Bukan Spam** | Memiliki beberapa link dan promosi moderat, namun peluangnya belum melampaui ambang 50%. |
| **E3** | 9 | 10 | +6.8242 | **0.9989 (99.89%)** | 🔴 **Spam** | Kombinasi link padat dan kata promosi masif mendorong peluang mendekati 100%. |
| **E4** | 2 | 11 | +0.7744 | **0.6845 (68.45%)** | 🔴 **Spam** | Meskipun tautan link sedikit, konsentrasi kata promosi sangat dominan (11 kata) sehingga peluang melewati ambang 50%. |

---

## 🧠 Bagian C: Pembahasan Pertanyaan Pemahaman Konsep

Pertanyaan pemahaman konsep dijawab secara komprehensif, akademis, dan terstruktur sesuai tinjauan teori machine learning:

### 1. Mengapa Logistic Regression termasuk klasifikasi padahal bernama "regression"? Apa fungsi Sigmoid?
**Jawaban Akademis:**  
Meskipun menyandang nama *"regression"*, Logistic Regression diklasifikasikan sebagai algoritma **klasifikasi** karena tujuan akhirnya adalah memprediksi probabilitas keanggotaan kelas diskrit (misalnya label biner 0 atau 1), bukan memproyeksikan kuantitas kontinu. Kata *"regression"* merujuk pada mekanisme matematis internalnya yang masih menyusun kombinasi linear antar bobot dan fitur input untuk mengestimasi logaritma rasio kemungkinan (*log-odds* atau *logit*): $z = b + \sum m_i x_i$.  
Peran krusial fungsi **sigmoid** ($\sigma(z) = \frac{1}{1 + e^{-z}}$) adalah bertindak sebagai fungsi aktivasi non-linear yang memetakan rentang linear tak terbatas $(-\infty, +\infty)$ menjadi interval probabilitas terstandarisasi $[0, 1]$. Transformasi ini memungkinkan output model diinterpretasikan secara sahih sebagai tingkat keyakinan bersyarat $P(Y=1 \mid X)$.

---

### 2. Perbedaan *Normal Equation* vs *Gradient Descent*, serta Kapan Memakai *Gradient Descent*?
**Jawaban Akademis:**  
* **Normal Equation** merupakan metode solusi analitik tertutup (*closed-form analytical solution*) yang menghitung vektor bobot optimal secara instan dalam satu langkah aljabar matriks $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. Metode ini bebas dari penentuan *hyperparameter learning rate* dan tidak memerlukan iterasi. Kelemahannya terletak pada kompleksitas komputasi inversi matriks sebesar $\mathcal{O}(d^3)$ ($d$ = jumlah fitur), yang sangat berat jika fiturnya masif.
* **Gradient Descent** merupakan metode optimasi numerik iteratif yang memperbarui bobot secara gradual menuruni vektor gradien loss: $\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \nabla L$.
* **Kapan Menggunakan Gradient Descent:**  
  Gradient Descent mutlak digunakan ketika:
  1. Jumlah fitur ($d$) sangat besar (ribuan hingga jutaan variabel, seperti pemrosesan teks atau citra) di mana operasi $(\mathbf{X}^T\mathbf{X})^{-1}$ mengalami *out-of-memory*.
  2. Jumlah baris observasi ($n$) sangat masif yang lebih efisien diproses secara batch/mini-batch.
  3. Matriks fitur bersifat singular/non-invertible akibat adanya multikolinearitas sempurna antar variabel independen.

---

### 3. Mengapa Evaluasi Dilakukan pada Data Testing, dan Apa Definisi *Overfitting*?
**Jawaban Akademis:**  
Evaluasi model machine learning wajib dilakukan pada **data testing** karena hakikat sejati dari sistem cerdas adalah kemampuan **generalisasi**, yaitu mengukur kecakapan model dalam menghasilkan prediksi akurat terhadap data baru yang belum pernah dilihat sebelumnya dalam skenario operasional riil. Jika evaluasi hanya dilakukan pada data latih, kita tidak dapat mendeteksi apakah model benar-benar memahami pola data fundamental atau hanya menghafal sampel latih.  
**Overfitting** adalah kegagalan pemodelan di mana algoritma terlalu kompleks sehingga mempelajari detail derau (*noise*), fluktuasi acak, dan kebetulan statistik pada data training secara berlebihan. Gejala khas overfitting ditandai dengan error yang sangat mendekati nol pada data training, namun akurasinya jatuh drastis ketika diuji pada data testing (*high variance, poor generalization*).

---

### 4. Mengapa Linear Regression Menggunakan MSE, sedangkan Logistic Regression Menggunakan *Log Loss*?
**Jawaban Akademis:**  
Pemilihan fungsi loss diturunkan secara fundamental dari asumsi distribusi probabilitas target melalui prinsip *Maximum Likelihood Estimation* (MLE):
* Pada **Linear Regression**, variabel target kontinu diasumsikan memiliki galat (*residual*) yang berdistribusi normal Gauss. Prinsip MLE pada distribusi Gaussian secara matematis menghasilkan minimasi jumlah kuadrat galat, yaitu **MSE (Mean Squared Error)**. Kurva fungsi loss MSE berbentuk mangkuk cembung murni (*strictly convex*), menjamin adanya satu titik minimum global yang mudah dicapai oleh solver analitik maupun gradien.
* Pada **Logistic Regression**, target output adalah probabilitas biner diskrit yang mengikuti distribusi Bernoulli. Jika MSE diterapkan pada kombinasi linear yang telah dilewatkan fungsi non-linear sigmoid, fungsi objektif yang dihasilkan akan berbentuk **non-convex** (bergelombang dengan banyak jebakan titik minimum lokal / *local minima*). Oleh karena itu, penurunan MLE pada Bernoulli menghasilkan **Log Loss (Cross-Entropy Loss)**. Selain menjamin sifat fungsi tetap *convex*, Log Loss memberikan penalti logaritmik yang sangat ekstrem (mendekati tak hingga) jika model memberikan probabilitas tinggi untuk kelas yang salah, sehingga memaksa gradien memperbarui bobot secara agresif.

---

### 5. Mengapa Angka Evaluasi Antar Mahasiswa Berbeda Walaupun Dataset Sama?
**Jawaban Akademis:**  
Variasi angka evaluasi akhir (koefisien, intercept, akurasi, presisi) antar mahasiswa terjadi akibat ketentuan penerapan **`random_state` yang disesuaikan dengan 2 digit terakhir NIM masing-masing** (untuk NIM `241011402689`, digunakan `random_state = 89`).  
Parameter `random_state` mengontrol seed generator bilangan acak (*pseudo-random number generator*) pada fungsi `train_test_split`. Perbedaan seed menghasilkan permutasi indeks baris data yang berbeda saat pemisahan set data latihan dan set data pengujian. Karena komposisi titik data latih yang dipelajari algoritma berbeda untuk setiap mahasiswa, gradien optimasi dan parameter garis keputusan akhir secara alamiah menghasilkan angka yang unik, kendati algoritma komputasi dan dataset dasarnya sama persis.

---

## 🏆 Bagian Bonus (+5): Uji Validasi Komparatif dengan Scikit-Learn

Untuk membuktikan keabsahan kode mandiri dari nol (*from scratch*), dilakukan validasi silang terhadap pustaka baku industri `scikit-learn`:

### 1. Validasi Model Linear Regression
Model NumPy mandiri diuji tanding dengan `sklearn.linear_model.LinearRegression`:

| Komponen Parameter & Metrik | Model Mandiri (*Normal Equation*) | Scikit-Learn (`LinearRegression`) | Tingkat Kesesuaian |
|---|:---:|:---:|:---:|
| **Intercept ($b$)** | `27.557043` | `27.557043` | **Identik 100% (Presisi 6 Desimal)** |
| **Koefisien ($m$)** | `9.176249` | `9.176249` | **Identik 100% (Presisi 6 Desimal)** |
| **Testing MSE** | `37.679383` | `37.679383` | **Identik 100% (Presisi 6 Desimal)** |
| **Testing RMSE** | `6.138353` | `6.138353` | **Identik 100% (Presisi 6 Desimal)** |
| **Testing $R^2$ Score** | `0.924637` | `0.924637` | **Identik 100% (Presisi 6 Desimal)** |

*Hasil komparasi membuktikan bahwa modul Linear Regression mandiri bebas dari cacat logika matematika dan menghasilkan aproksimasi analitik yang sempurna.*

---

### 2. Validasi Model Logistic Regression
Model NumPy mandiri diuji tanding dengan Scikit-Learn dalam dua konfigurasi regularisasi:

| Parameter & Metrik Testing | Model Mandiri (GD 10k iterasi) | Scikit-Learn (Tanpa Reg: `C=np.inf`) | Scikit-Learn (Default L2: `C=1.0`) |
|---|:---:|:---:|:---:|
| **Intercept ($b$)** | `-12.5226` | `-15.1499` | `-11.9559` |
| **Koefisien $m_1$ (`JumlahLink`)** | `1.0106` | `1.2351` | `0.9538` |
| **Koefisien $m_2$ (`KataPromo`)** | `1.0251` | `1.2364` | `0.9776` |
| **Testing Accuracy** | **0.7333 (73.33%)** | **0.7333 (73.33%)** | **0.7333 (73.33%)** |
| **Testing Precision** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** |
| **Testing Recall** | **0.6364 (63.64%)** | **0.6364 (63.64%)** | **0.6364 (63.64%)** |
| **Testing F1-Score** | **0.7778 (77.78%)** | **0.7778 (77.78%)** | **0.7778 (77.78%)** |

* **Penjelasan Ilmiah Perbedaan Koefisien:**
  1. **Pengaruh Penalti Regularisasi ($L_2$ Ridge Penalty):** Konfigurasi default Scikit-Learn mengaktifkan regularisasi $L_2$ (`C = 1.0`), yang memberikan penalti kuadratik terhadap besaran magnitudo bobot ($\lambda \|\mathbf{w}\|^2$). Penalti ini memicu fenomena penyusutan bobot (*weight shrinkage*), sehingga koefisien Scikit-Learn menyusut ke angka 0.95–0.97. Sedangkan model buatan sendiri dirancang tanpa penalti regularisasi murni.
  2. **Dinamika Algoritma Optimizer:** Pada mode tanpa regularisasi (`C = np.inf`), Scikit-Learn memanfaatkan algoritma quasi-Newton orde-dua canggih (*L-BFGS*) yang mampu mencari titik stasioner absolut. Sementara implementasi mandiri menggunakan *First-Order Gradient Descent* 10.000 iterasi dengan laju pembelajaran tetap yang laju penurunannya melambat di ujung iterasi.
  3. **Konsistensi Metrik Prediktif:** Meskipun posisi sudut garis batas mengalami pergeseran mikro, **seluruh metrik evaluasi pada data testing menghasilkan angka yang persis sama (Akurasi 73.33%, Presisi 100%, F1-Score 77.78%)**. Ini menegaskan bahwa model mandiri memiliki kehandalan klasifikasi yang setara dengan pustaka Scikit-Learn.

---

## 🚀 Panduan Eksekusi & Reproduksibilitas

### 1. Kebutuhan Lingkungan (*Environment Prerequisites*)
Pastikan Python versi 3.8 atau yang lebih baru telah terpasang, beserta pustaka penunjang:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 2. Kloning Repositori & Menjalankan Notebook
```bash
# Kloning repositori GitHub
git clone https://github.com/aryaarach/uts-machine-learning-241011402689.git

# Masuk ke direktori proyek
cd uts-machine-learning-241011402689

# Jalankan server Jupyter Notebook
jupyter notebook tugas_UTS.ipynb
```

### 3. Eksekusi Penuh (*Reproducibility*)
Untuk mereproduksi seluruh komputasi, tabel metrik, dan visualisasi grafik secara berurutan:
* Buka file `tugas_UTS.ipynb`.
* Akses bilah menu atas: **Kernel $\rightarrow$ Restart & Run All**.
* Seluruh sel akan dieksekusi secara otomatis dari sel inisialisasi identitas hingga pengujian bonus.
