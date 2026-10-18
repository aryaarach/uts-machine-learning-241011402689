# UTS Machine Learning: Linear & Logistic Regression from Scratch

> Proyek Ujian Tengah Semester (UTS) mata kuliah **Machine Learning**. Seluruh model dan metrik evaluasi diimplementasikan **murni dari nol (*from scratch*) menggunakan NumPy**, tanpa bantuan library *black-box* seperti scikit-learn (scikit-learn hanya dipakai pada bagian bonus untuk validasi hasil).

---

## 📌 Identitas Mahasiswa

| Informasi | Keterangan |
|---|---|
| **Nama Lengkap** | Arya Rachman Indrawan |
| **NIM** | `241011402689` |
| **Kelas** | Reguler B (Reg B) |
| **Mata Kuliah** | Machine Learning |
| **Dua Digit Terakhir NIM** | `89` |
| **Nilai `random_state`** | `89` *(digunakan untuk pengacakan train-test split)* |

---

## 💡 Ringkasan Cepat (TL;DR untuk Bahan Presentasi / Tanya Jawab)

Jika dosen meminta ringkasan singkat dalam 1 menit:
1. **Bagian A (Linear Regression):** Kita memprediksi omzet penjualan berdasarkan biaya iklan bulanan. Didapat rumus $\hat{y} = 27.56 + 9.18X$. Artinya, tanpa pasang iklan pun penjualan dasar toko ada di angka **27,56 juta Rp**, dan setiap tambahan modal iklan **1 juta Rp** mendatangkan tambahan penjualan rata-rata **9,18 juta Rp** (ROI sangat sehat). Model ini sangat akurat dengan **$R^2 = 92.46\%$** pada data uji.
2. **Bagian B (Logistic Regression):** Kita mendeteksi email spam dari jumlah link dan kata promo. Model menghasilkan akurasi **73.33%** dengan **Presisi 100%** pada threshold 0.5. Artinya, **tidak ada satu pun email penting/asli yang salah masuk ke folder spam** (*zero false positive*).
3. **Validasi Scikit-Learn:** Model Linear Regression buatan sendiri menghasilkan angka yang **100% identik hingga 6 desimal** dengan `sklearn.linear_model.LinearRegression`.

---

## 📂 Struktur Repositori

```text
uts-machine-learning-241011402689/
├── README.md            # Dokumentasi lengkap & panduan penjelasan hasil
├── tugas_UTS.ipynb      # Notebook pengerjaan (sudah di-Run All lengkap output & grafik)
├── Ads_Data.csv         # Dataset Biaya Iklan vs Penjualan (40 baris data)
└── Email_Data.csv       # Dataset Jumlah Link & Kata Promo vs Spam (60 baris data)
```

---

## 📊 Bagian A: Linear Regression (`Ads_Data.csv`)

### 1. Data & Persamaan Garis Regresi
- **Dataset:** 40 data observasi toko online.
- **Split Data:** 80% Training (32 data) dan 20% Testing (8 data) dengan `random_state = 89`.
- **Metode:** *Normal Equation* (solusi analitis aljabar matriks $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$).

$$\hat{y} = 27.5570 + 9.1762 \cdot X$$

* **Arti Intercept ($b = 27.5570$):** Jika perusahaan sama sekali tidak mengeluarkan biaya iklan ($X = 0$), estimasi penjualan dasar bulanan adalah sekitar **27,56 juta Rp**.
* **Arti Slope / Koefisien ($m = 9.1762$):** Setiap penambahan anggaran iklan sebesar **1 juta Rp**, rata-rata nilai penjualan diprediksi naik sebesar **9,18 juta Rp**.

---

### 2. Evaluasi Model (Training vs Testing)

| Metrik Evaluasi | Data Training (32 data) | Data Testing (8 data) | Cara Menjelaskan Intinya |
|---|---|---|---|
| **MAE** *(Mean Absolute Error)* | **4.4018** | **5.0772** | Rata-rata selisih tebakan meleset sekitar 4–5 juta Rp dari realita. |
| **MSE** *(Mean Squared Error)* | **35.7863** | **37.6794** | Kuadrat rata-rata error (penalti buat error besar). |
| **RMSE** *(Root Mean Squared Error)* | **5.9822** | **6.1384** | Standar deviasi error sekitar 6,14 juta Rp. |
| **$R^2$ Score** *(Koefisien Determinasi)* | **0.9456 (94.56%)** | **0.9246 (92.46%)** | **92,46%** variasi penjualan sukses dijelaskan oleh variabel biaya iklan. |

> **Apakah model ini bagus?**  
> **Sangat bagus dan tidak overfitting.** Performa di data training ($R^2 = 94.56\%$) dan data testing ($R^2 = 92.46\%$) memiliki selisih yang sangat tipis (hanya beda ~2%). Ini bukti bahwa model mampu menggeneralisasi data baru dengan sangat stabil, bukan sekadar menghafal data latihan.

---

### 3. Eksperimen Learning Rate pada Gradient Descent

Model dilatih ulang menggunakan metode iteratif *Gradient Descent* sebanyak **5.000 iterasi** dengan 3 variasi *learning rate* ($\alpha$):

| Learning Rate ($\alpha$) | Intercept ($b$) | Koefisien ($m$) | Loss Akhir (MSE) | Perilaku & Analogi |
|---|---|---|---|---|
| **`0.001` (Terlalu Kecil)** | 18.4763 | 10.5492 | 53.26 | **Terlalu Lambat (*Under-shooting*):** Langkahnya terlalu pelan seperti melangkah jengkal demi jengkal. Di iterasi ke-5000 dia masih belum sampai ke titik dasar lembah. |
| **`0.02` (Ideal / Optimal)** | **27.5570** | **9.1762** | **35.79** | **Konvergen Sempurna:** Ukuran langkah pas. Hasilnya **identik persis** dengan solusi rumus matematis *Normal Equation*. |
| **`0.1` (Terlalu Besar)** | `NaN` | `NaN` | `overflow` | **Gagal Konvergen (*Divergen*):** Langkahnya kegedean sampai melompati titik terendah. Akibatnya nilai error melambung liar hingga komputer mengalami *overflow* (sekitar iterasi ke-370). |

---

### 4. Prediksi & Bahaya Ekstrapolasi

Prediksi penjualan dengan model regresi:
1. **Biaya Iklan 4,5 juta Rp** $\rightarrow$ Prediksi Penjualan = **68,85 juta Rp** *(Interpolasi - Aman & Terpercaya)*
2. **Biaya Iklan 7,0 juta Rp** $\rightarrow$ Prediksi Penjualan = **91,79 juta Rp** *(Interpolasi - Aman & Terpercaya)*
3. **Biaya Iklan 12,0 juta Rp** $\rightarrow$ Prediksi Penjualan = **137,67 juta Rp** *(Ekstrapolasi - Perlu Hati-hati!)*

> **Mengapa prediksi 12 juta kurang bisa dipercaya 100%?**  
> Data latih asli kita hanya berada di rentang **0,8 sampai 9,9 juta Rp**. Angka 12 juta berada jauh di luar jangkauan data historis (*ekstrapolasi*). Dalam bisnis riil, berlaku hukum *diminishing returns* (kejenuhan pasar): menambah budget iklan terus-menerus belum tentu menaikkan omzet secara garis lurus, bisa saja kurvanya melandai karena pasar sudah jenuh.

---

## 📧 Bagian B: Logistic Regression (`Email_Data.csv`)

### 1. Data & Persamaan Model
- **Dataset:** 60 email dengan fitur `JumlahLink` ($x_1$) dan `KataPromo` ($x_2$), serta target `Spam` (1 = Spam, 0 = Bukan Spam).
- **Split Data:** 75% Training (45 data) dan 25% Testing (15 data) dengan `random_state = 89`.
- **Optimasi:** *Gradient Descent* ($\alpha = 0.1$, 10.000 iterasi). Log loss berhasil ditekan turun dari **0.6931** menjadi **0.1657**.

$$\text{Logit: } z = -12.5226 + 1.0106 \cdot \text{JumlahLink} + 1.0251 \cdot \text{KataPromo}$$

$$P(\text{Spam}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Garis Batas Keputusan (*Decision Boundary* pada $z = 0$):**
  $$\text{KataPromo} = 12.2164 - 0.9859 \cdot \text{JumlahLink}$$
* **Artinya:** Kedua fitur sama-sama memberikan sinyal kuat ke arah spam (koefisien positif sekitar $+1.01$ dan $+1.02$). Makin banyak link dan kata promosi di dalam email, kurva sigmoid akan mendorong nilai probabilitas mendekati angka 1 (Spam).

---

### 2. Evaluasi Data Testing (Ambang Batas Default 0.5)

Dari **15 data testing**:
- Email Bukan Spam (Kelas 0): 4 email
- Email Spam (Kelas 1): 11 email

| Confusion Matrix | Prediksi: Bukan Spam (0) | Prediksi: Spam (1) | Keterangan Lapangan |
|---|:---:|:---:|---|
| **Aktual: Bukan Spam (0)** | **TN = 4** | **FP = 0** | **Sempurna!** Nol email biasa yang terbuang ke spam. |
| **Aktual: Spam (1)** | **FN = 4** | **TP = 7** | Ada 4 email spam licik yang lolos masuk ke inbox. |

| Metrik | Skor | Cara Menjelaskan Maknanya |
|---|:---:|---|
| **Akurasi** | **73.33%** | Dari 15 email uji, 11 email ditebak dengan benar ($11/15$). |
| **Presisi (*Precision*)** | **100.0%** | Sekalinya model memvonis email itu "Spam", tebakannya **pasti 100% akurat** ($7/7$). Tidak ada korban salah tangkap. |
| **Recall / Sensitivitas** | **63.64%** | Dari total 11 email spam asli, model berhasil menyaring 7 email ($7/11$). Sisanya 4 email lolos. |
| **F1-Score** | **77.78%** | Nilai rata-rata harmonik antara presisi dan recall. |

---

### 3. Eksperimen Threshold: Kenapa Memilih 0.5?

Di dunia nyata, kita bisa menggeser ambang batas (*threshold*) probabilitas untuk menentukan kapan sebuah email dianggap spam:

| Threshold | Confusion Matrix `[[TN, FP], [FN, TP]]` | Precision | Recall | Analisis Risiko Pengguna |
|:---:|:---:|:---:|:---:|---|
| **`0.3`** *(Sensitif)* | `[[3, 1], [2, 9]]` | 90.00% | **81.82%** | Spam lebih banyak tertangkap, **tapi muncul 1 False Positive** (ada email penting nasabah/kampus nyasar ke folder spam). Berbahaya! |
| **`0.5`** *(Ideal / Dipilih)* | `[[4, 0], [4, 7]]` | **100.0%** | **63.64%** | **FP = 0**. Keamanan inbox terjaga penuh: email penting dijamin tidak akan terbuang, sementara sebagian besar spam tetap terfilter. |
| **`0.7`** *(Ketat)* | `[[4, 0], [6, 5]]` | **100.0%** | **45.45%** | FP tetap 0, tapi Recall anjlok parah (lebih dari separuh email spam jebol masuk inbox). Kurang efektif. |

> **Kesimpulan:** Threshold **0.5 adalah titik paling aman dan optimal**. Dalam sistem email, *False Positive* (email penting hilang/tak terbaca) memiliki konsekuensi yang jauh lebih fatal bagi user dibandingkan *False Negative* (hanya perlu menghapus manual sedikit spam di inbox).

---

### 4. Prediksi Email Baru

Menguji 4 sampel email baru dengan threshold 0.5:

| ID Email | Jumlah Link | Kata Promo | Probabilitas Spam $P(\text{Spam})$ | Keputusan Model | Penjelasan Logika |
|:---:|:---:|:---:|:---:|:---:|---|
| **E1** | 1 | 2 | **0.0001 (0.01%)** | 🟢 **Bukan Spam** | Link dan kata promo sangat minim. Sangat bersih. |
| **E2** | 5 | 6 | **0.2109 (21.09%)** | 🟢 **Bukan Spam** | Ada beberapa link & promo, tapi belum menembus ambang batas 50%. |
| **E3** | 9 | 10 | **0.9989 (99.89%)** | 🔴 **Spam** | Link bertaburan dan penuh kata rayuan promo. Hampir pasti 100% spam. |
| **E4** | 2 | 11 | **0.6844 (68.44%)** | 🔴 **Spam** | Meski link sedikit, indikasi kata promonya sangat mencolok (11 kata), melewati ambang 50%. |

---

## 🧠 Bagian C: Cheat Sheet 5 Konsep Machine Learning

Bagian ini dirancang agar mudah dijelaskan secara lisan saat ditanya dosen:

#### 1. Kenapa namanya *Logistic Regression* padahal fungsinya untuk *Klasifikasi*?
> **Inti Jawaban:**  
> Karena di balik layarnya, ia sebenarnya masih melakukan perhitungan **regresi linear** terhadap nilai *log-odds* ($z = b + \sum m_i x_i$). Hanya saja, angka linear yang bisa minus atau plus tak terhingga tersebut tidak langsung dipakai sebagai output, melainkan **diumpan ke fungsi Sigmoid** ($\sigma(z) = \frac{1}{1 + e^{-z}}$). Fungsi sigmoid memampatkan (*squeeze*) nilai apa pun menjadi skala probabilitas antara **0 sampai 1**. Karena output akhirnya berupa probabilitas untuk menentukan kelas label (0 atau 1), maka tugasnya adalah klasifikasi.

#### 2. Apa beda *Normal Equation* vs *Gradient Descent*, dan kapan harus pakai Gradient Descent?
> **Inti Jawaban:**  
> - **Normal Equation:** Cara instan sekali hitung menggunakan rumus matriks invers $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. Tidak perlu repot atur learning rate atau tunggu iterasi, tapi boros memori dan lambat jika fiturnya banyak karena operasi invers matriks berskala $\mathcal{O}(d^3)$.
> - **Gradient Descent:** Cara bertahap (iteratif) menuruni lereng bukit error sampai menemukan titik terendah.
> - **Kapan pakai Gradient Descent?** Ketika data berukuran besar (ratusan ribu hingga jutaan baris/fitur) di mana RAM komputer tidak sanggup lagi membalik matriks, atau ketika matriks fitur tidak bisa diinvers (*singular matrix*).

#### 3. Kenapa wajib evaluasi di *Data Testing*? Apa itu *Overfitting*?
> **Inti Jawaban:**  
> - Mesin belajar bertujuan agar bisa pintar di dunia nyata (**kemampuan generalisasi**), bukan sekadar jago di soal latihan. Menguji di data training sama saja seperti memberi siswa ujian dengan soal yang kunci jawabannya sudah dihafal sebelumnya.
> - **Overfitting** adalah kondisi di mana model "terlalu hafal" data latihan sampai kebisingan (*noise*) ikut dipelajari. Akibatnya, nilainya sempurna di data training, tapi saat diberi data baru (data testing) hasilnya hancur.

#### 4. Kenapa Linear Regression pakai MSE, sedangkan Logistic Regression pakai Log Loss?
> **Inti Jawaban:**  
> - **Linear Regression** menebak angka kontinu. Selisih tebakan dengan target diasumsikan mengikuti kurva lonceng normal (Gaussian), sehingga rumus paling cocok secara matematis adalah kuadrat selisih (**MSE**). Kurva MSE pada regresi linear berbentuk mangkok cembung (*convex*), jadi mudah dicari titik dasarnya.
> - **Logistic Regression** menebak probabilitas kejadian biner (distribusi Bernoulli). Jika kita memaksakan rumus MSE pada fungsi sigmoid yang berbelok-belok, kurva errornya akan bergelombang (*non-convex*) dan penuh jebakan lokal (*local minima*). Oleh karena itu, digunakan **Log Loss (Cross-Entropy)** yang menjamin kurva tetap cembung (*convex*) dan memberikan hukuman penalti sangat telak jika model overpede menebak salah.

#### 5. Kenapa hasil angka tiap mahasiswa bisa berbeda meski dataset dan rumusnya sama?
> **Inti Jawaban:**  
> Karena ada parameter **`random_state` yang disetel sesuai 2 digit terakhir NIM masing-masing**. Nilai acak ini menentukan baris data mana saja yang terpilih masuk ke kantong *training* dan kantong *testing* saat fungsi `train_test_split` dijalankan. Karena data latihan yang disuapkan ke algoritma tiap mahasiswa sedikit berlainan, titik keseimbangan parameter akhir ($b$ dan $m$) otomatis akan menghasilkan angka yang unik.

---

## 🏆 Bonus (+5): Uji Validasi dengan Scikit-Learn

Untuk membuktikan bahwa algoritma buatan sendiri dari materi kuliah sudah benar dan tidak ada *bug*, kita uji bandingkan langsung dengan library standar industri `scikit-learn`:

### 1. Linear Regression
| Parameter / Metrik | Model Sendiri (*Normal Equation*) | `sklearn.linear_model.LinearRegression` | Status |
|---|:---:|:---:|:---:|
| **Intercept ($b$)** | `27.557043` | `27.557043` | **Identik 100%** |
| **Koefisien ($m$)** | `9.176249` | `9.176249` | **Identik 100%** |
| **Testing MSE** | `37.679383` | `37.679383` | **Identik 100%** |
| **Testing RMSE** | `6.138353` | `6.138353` | **Identik 100%** |
| **Testing $R^2$** | `0.924637` | `0.924637` | **Identik 100%** |

### 2. Logistic Regression
| Parameter / Metrik | Model Sendiri (GD 10k iterasi) | Scikit-Learn (Tanpa Reg, `C=inf`) | Scikit-Learn (Default $L_2$, `C=1.0`) |
|---|:---:|:---:|:---:|
| **Intercept ($b$)** | `-12.5226` | `-15.1499` | `-11.9559` |
| **Koef. $m_1$ (Link)** | `1.0106` | `1.2351` | `0.9538` |
| **Koef. $m_2$ (Promo)** | `1.0251` | `1.2364` | `0.9776` |
| **Testing Accuracy** | **0.7333 (73.33%)** | **0.7333 (73.33%)** | **0.7333 (73.33%)** |
| **Testing Precision** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** |
| **Testing Recall** | **0.6364 (63.64%)** | **0.6364 (63.64%)** | **0.6364 (63.64%)** |
| **Testing F1-Score** | **0.7778 (77.78%)** | **0.7778 (77.78%)** | **0.7778 (77.78%)** |

> **Mengapa koefisien Logistic Regression ada selisih sedikit?**
> 1. **Faktor Regularisasi ($L_2$):** Secara *default*, Scikit-Learn mengaktifkan penalti regularisasi L2 (`C=1.0`) untuk mencegah bobot terlalu besar (*weight shrinkage*). Itu sebabnya koefisien default sklearn menyusut ke angka 0.95–0.97. Sedangkan model buatan kita murni tanpa regularisasi.
> 2. **Perbedaan Solver / Optimizer:** Pada versi tanpa regularisasi (`C=inf`), Scikit-Learn menggunakan algoritma optimasi tingkat lanjut (*L-BFGS*) yang mampu melangkah sampai ke titik puncak absolut, sedangkan model kita memakai Gradient Descent sederhana 10.000 iterasi.
> 3. **Keputusan Prediksi Tetap Sama:** Kendati posisi garis batas bergeser sedikit milimeter, **semua metrik evaluasi pada data testing menghasilkan nilai yang persis sama (Akurasi 73.33%, Presisi 100%, F1 77.78%)**.

---

## 🚀 Cara Menjalankan Notebook

Pastikan Python 3 sudah terpasang beserta pustaka data science standar:

```bash
pip install numpy pandas matplotlib scikit-learn
```

1. Clone repositori ini atau download folder tugas:
   ```bash
   git clone https://github.com/aryaarach/uts-machine-learning-241011402689.git
   cd uts-machine-learning-241011402689
   ```
2. Pastikan file `Ads_Data.csv` dan `Email_Data.csv` berada satu folder dengan `tugas_UTS.ipynb`.
3. Buka notebook menggunakan Jupyter Notebook / VS Code:
   ```bash
   jupyter notebook tugas_UTS.ipynb
   ```
4. Pilih menu **Kernel $\rightarrow$ Restart & Run All** untuk mengeksekusi seluruh sel secara berurutan.
