# Prediksi Keberhasilan Kampanye Telemarketing Bank dengan Machine Learning

## Ringkasan Proyek

Proyek ini berfokus pada **optimalisasi kampanye telemarketing** sebuah bank di Portugal yang bertujuan untuk menawarkan produk deposito berjangka (*term deposit*). Tantangan utama kampanye ini adalah tingginya biaya operasional yang disebabkan oleh durasi panggilan yang panjang, terutama ketika kampanye dilakukan dalam skala besar. Tidak semua interaksi telepon berujung pada keberhasilan (pembukaan deposito), sehingga banyak waktu dan sumber daya yang terbuang sia-sia.

Untuk mengatasi masalah ini, proyek ini mengembangkan **model klasifikasi machine learning** yang mampu memprediksi apakah seorang nasabah akan membuka deposito atau tidak **sebelum panggilan dilakukan**. Dengan model ini, bank dapat memprioritaskan nasabah yang paling potensial, menerapkan strategi **batas waktu panggilan (cut-off)** yang efisien, dan pada akhirnya **menghemat biaya operasional secara signifikan** tanpa mengorbankan jangkauan nasabah.

Proyek ini mencakup seluruh siklus *data science*, mulai dari pemahaman masalah bisnis, analisis data eksploratif (EDA), *preprocessing* data, pemodelan, evaluasi metrik, hingga rekomendasi implementasi bisnis. Fokus utama evaluasi model adalah **Recall**, untuk memastikan bank tidak kehilangan nasabah potensial, sambil tetap menyeimbangkan biaya melalui analisis **Precision** dan **F1-Score**.

---

## Daftar Isi

1.  [Pemahaman Masalah Bisnis](#pemahaman-masalah-bisnis)
    -   [Konteks Bisnis](#konteks-bisnis)
    -   [Pernyataan Masalah](#pernyataan-masalah)
    -   [Tujuan Proyek](#tujuan-proyek)
    -   [Pendekatan Analitis](#pendekatan-analitis)
2.  [Strategi Evaluasi Model](#strategi-evaluasi-model)
    -   [Matriks Konfusi dan Analisis Error](#matriks-konfusi-dan-analisis-error)
    -   [Pemilihan Metrik Utama](#pemilihan-metrik-utama)
3.  [Pemahaman Data (Data Understanding)](#pemahaman-data-data-understanding)
    -   [Sumber Data](#sumber-data)
    -   [Kamus Data (Attribute Information)](#kamus-data-attribute-information)
    -   [Karakteristik Dataset](#karakteristik-dataset)
4.  [Preprocessing dan Analisis Data Eksploratif (EDA)](#preprocessing-dan-analisis-data-eksploratif-eda)
    -   [Penanganan Data Duplikat](#penanganan-data-duplikat)
    -   [Penanganan Nilai Hilang ("Unknown")](#penanganan-nilai-hilang-unknown)
    -   [Transformasi Fitur](#transformasi-fitur)
    -   [Wawasan dari EDA](#wawasan-dari-eda)
5.  [Persiapan Data untuk Pemodelan](#persiapan-data-untuk-pemodelan)
    -   [Pencegahan Kebocoran Data (Data Leakage)](#pencegahan-kebocoran-data-data-leakage)
    -   [Encoding Fitur Kategorikal](#encoding-fitur-kategorikal)
    -   [Pemisahan Data (Train-Test Split)](#pemisahan-data-train-test-split)
6.  [Pemodelan dan Evaluasi](#pemodelan-dan-evaluasi)
    -   [Benchmarking Model Awal](#benchmarking-model-awal)
    -   [Penanganan Ketidakseimbangan Kelas (Class Imbalance)](#penanganan-ketidakseimbangan-kelas-class-imbalance)
    -   [Hyperparameter Tuning](#hyperparameter-tuning)
    -   [Analisis Feature Importance](#analisis-feature-importance)
7.  [Kesimpulan dan Rekomendasi](#kesimpulan-dan-rekomendasi)
    -   [Ringkasan Hasil](#ringkasan-hasil)
    -   [Rekomendasi Bisnis](#rekomendasi-bisnis)
    -   [Implementasi Model](#implementasi-model)
    -   [Limitasi Model](#limitasi-model)

---

## Pemahaman Masalah Bisnis

### Konteks Bisnis
Sebuah institusi perbankan di Portugal sedang gencar melakukan kampanye pemasaran untuk produk **deposito berjangka** melalui telemarketing. Dalam kampanye ini, agen menghubungi nasabah satu per satu melalui panggilan telepon untuk menawarkan produk tersebut. Meskipun telemarketing efektif untuk menjangkau seluruh basis nasabah, metode ini memiliki tantangan besar dari sisi efisiensi biaya.

Tantangan utamanya adalah **tingginya biaya operasional** yang timbul dari durasi panggilan yang panjang. Ketika dilakukan dalam skala besar, akumulasi durasi panggilan menjadi beban finansial yang signifikan. Masalah ini diperparah oleh fakta bahwa tidak semua panggilan telepon menghasilkan konversi (pembukaan deposito), yang berarti banyak waktu, tenaga agen, dan biaya pulsa yang terbuang untuk nasabah yang sejak awal tidak tertarik.

Selama kampanye, berbagai data penting dikumpulkan, seperti profil demografis nasabah (usia, pekerjaan, status perkawinan), riwayat interaksi dengan kampanye sebelumnya, serta data spesifik panggilan seperti durasi (`duration`). Dari analisis awal, variabel `duration` memiliki korelasi yang sangat kuat dengan hasil akhir kampanye (`y`), yang menandakan bahwa durasi percakapan adalah indikator penting dari minat nasabah.

Perusahaan ingin memanfaatkan data historis ini untuk menyusun strategi efisiensi tanpa harus mengurangi jangkauan nasabah. Mereka ingin tahu bagaimana durasi percakapan bisa dikelola secara optimal agar:
1.  **Tidak membuang biaya** untuk nasabah yang jelas-jelas tidak tertarik.
2.  **Tetap mengoptimalkan peluang konversi** pada nasabah yang menunjukkan potensi untuk membuka deposito.

### Pernyataan Masalah
Bank ingin tetap **menghubungi seluruh nasabah** dalam kampanyenya, namun dihadapkan pada risiko inefisiensi yang signifikan:
- **Durasi Panggilan yang Terlalu Panjang:** Agen menghabiskan waktu berharga pada nasabah yang pada akhirnya menolak tawaran.
- **Pemborosan Biaya Komunikasi:** Setiap detik panggilan memiliki biaya, dan panggilan yang tidak produktif secara langsung meningkatkan beban operasional.
- **Beban Waktu Agen:** Waktu yang terbuang bisa dialokasikan untuk menghubungi nasabah lain yang lebih potensial.

Oleh karena itu, bank perlu menerapkan sebuah strategi **batas waktu maksimal percakapan (`duration_cut_off`)** yang cerdas dan berbasis data. Batas waktu ini harus optimal, artinya cukup singkat untuk memotong kerugian pada nasabah yang tidak prospektif, namun cukup panjang untuk tidak memotong percakapan dengan nasabah potensial yang sedang dalam proses konversi.

Berdasarkan analisis data historis yang dilakukan, potensi penghematan dari strategi ini sangat besar:
- **Penghematan Biaya:**
  - Total biaya aktual (tanpa *cut-off*): **€45.359,00**
  - Estimasi total biaya setelah *cut-off*: **€24.857,60**
  - **Potensi Penghematan: €20.501,40 (45%)**

- **Penghematan Waktu:**
  - Total durasi aktual: 10.638.243 detik (**177.304 menit**)
  - Estimasi total durasi setelah *cut-off*: 5.991.325 detik (**99.855 menit**)
  - **Potensi Penghematan: 4.646.918 detik (77.448 menit atau 43,68%)**

Penghematan waktu ini setara dengan **1.290 jam telepon**, atau hampir **161 hari kerja agen** (dengan asumsi 8 jam kerja per hari).

### Tujuan Proyek
Proyek ini bertujuan untuk mengatasi tantangan efisiensi tersebut dengan beberapa sasaran utama:
1.  **Meningkatkan Efisiensi Kampanye Telemarketing:** Mengurangi biaya dan waktu yang terbuang tanpa mengorbankan jangkauan total nasabah.
2.  **Menetapkan Batas Durasi Optimal (`duration_cut_off`):** Menggunakan data historis untuk menentukan batas waktu panggilan yang paling efisien.
3.  **Menghemat Biaya dan Waktu:** Secara kuantitatif mengurangi biaya operasional dan total durasi panggilan, terutama pada interaksi dengan nasabah yang tidak tertarik.
4.  **Membangun Model Prediktif:** Mengembangkan model *machine learning* yang akurat untuk mengidentifikasi nasabah yang paling mungkin membuka deposito (`y = "yes"`) sebelum panggilan dilakukan.

### Pendekatan Analitis
Untuk mencapai tujuan tersebut, pendekatan analitis yang sistematis diterapkan:
1.  **Analisis Data Eksploratif (EDA):** Menganalisis variabel-variabel yang ada untuk menemukan pola dan korelasi dengan variabel target (`y`). Ini termasuk memahami profil nasabah yang cenderung menerima dan menolak tawaran.
2.  **Feature Engineering:** Menambahkan fitur-fitur baru seperti `call_fee` (estimasi biaya panggilan) dan `duration_cut_off` untuk mendukung analisis efisiensi.
3.  **Pemodelan Machine Learning:** Menerapkan beberapa model klasifikasi untuk memprediksi probabilitas seorang nasabah akan membuka deposito. Model ini akan menjadi dasar bagi tim telemarketing dalam memprioritaskan panggilan.
4.  **Evaluasi Strategi:** Mengevaluasi dampak dari penerapan strategi *cut-off* dalam menurunkan total biaya dan durasi panggilan berdasarkan hasil prediksi model.

---

## Strategi Evaluasi Model

Pemilihan metrik evaluasi yang tepat adalah kunci keberhasilan proyek ini, karena akan menentukan bagaimana performa model diukur dan dioptimalkan sesuai dengan tujuan bisnis.

### Matriks Konfusi dan Analisis Error

Karena ini adalah masalah klasifikasi biner (nasabah "membuka deposito" atau "tidak membuka deposito"), matriks konfusi menjadi alat evaluasi utama.

|                                     | **PREDIKSI: Tidak Buka (0)** | **PREDIKSI: Buka (1)** |
| ----------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **AKTUAL: Tidak Buka (0)** | **True Negative (TN)**<br>Model memprediksi nasabah tidak tertarik, dan prediksi itu benar.   | **False Positive (FP) / Error Tipe 1**<br>Model memprediksi nasabah tertarik, padahal tidak. |
| **AKTUAL: Buka (1)** | **False Negative (FN) / Error Tipe 2**<br>Model memprediksi nasabah tidak tertarik, padahal iya. | **True Positive (TP)**<br>Model memprediksi nasabah tertarik, dan prediksi itu benar.   |

#### Analisis Konsekuensi Error:
- **Error Tipe 1 (False Positive):**
  - **Masalah:** Model mengatakan nasabah akan tertarik, tetapi kenyataannya tidak.
  - **Konsekuensi Bisnis:** **Pemborosan sumber daya.** Bank akan menghabiskan waktu agen dan biaya telepon untuk menghubungi nasabah yang tidak akan pernah melakukan konversi. Ini adalah masalah efisiensi yang ingin dipecahkan.

- **Error Tipe 2 (False Negative):**
  - **Masalah:** Model mengatakan nasabah tidak akan tertarik, padahal sebenarnya mereka adalah prospek potensial.
  - **Konsekuensi Bisnis:** **Kehilangan peluang pendapatan.** Bank akan melewatkan nasabah yang seharusnya bisa membuka deposito. Ini adalah kerugian dari sisi pertumbuhan bisnis.

### Pemilihan Metrik Utama

Berdasarkan analisis *trade-off* di atas, kita tidak ingin terlalu banyak *False Positives* (untuk menekan biaya), tetapi juga tidak boleh terlalu banyak *False Negatives* (agar tidak kehilangan prospek). Oleh karena itu, model yang ideal harus seimbang.

Metrik evaluasi yang akan digunakan adalah:
1.  **Recall (Sensitivity):**
    - **Rumus:** $Recall = \frac{TP}{TP + FN}$
    - **Fokus:** Mengukur kemampuan model untuk mengidentifikasi **semua** nasabah yang sebenarnya tertarik (kelas positif).
    - **Relevansi Bisnis:** Recall tinggi sangat penting karena **kehilangan nasabah potensial (FN) lebih merugikan daripada membuang biaya untuk panggilan yang tidak berhasil (FP)**. Kita ingin meminimalkan peluang yang terlewatkan. Oleh karena itu, **Recall menjadi metrik optimasi utama**.

2.  **Precision:**
    - **Rumus:** $Precision = \frac{TP}{TP + FP}$
    - **Fokus:** Dari semua nasabah yang diprediksi tertarik, berapa persen yang benar-benar tertarik.
    - **Relevansi Bisnis:** Precision tinggi membantu menekan biaya dengan memastikan bahwa panggilan yang direkomendasikan oleh model memiliki tingkat keberhasilan yang tinggi.

3.  **F1-Score:**
    - **Rumus:** $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
    - **Fokus:** Memberikan keseimbangan antara Precision dan Recall. Metrik ini berguna untuk mendapatkan gambaran umum performa model yang seimbang.

4.  **PR Curve & ROC AUC:**
    - **Relevansi:** Dataset ini bersifat **tidak seimbang (imbalanced)**, di mana jumlah nasabah yang menolak (`"no"`) jauh lebih banyak daripada yang menerima (`"yes"`). Dalam kondisi seperti ini, metrik akurasi bisa menyesatkan. Kurva ROC (Receiver Operating Characteristic) dan PR (Precision-Recall) memberikan gambaran yang lebih baik tentang performa model di berbagai ambang batas probabilitas.

**Kesimpulan Strategi:** Model akan dilatih dan dioptimalkan untuk memaksimalkan **Recall**, sambil tetap memantau **Precision** untuk memastikan biaya tidak membengkak secara tidak terkendali.

---

## Pemahaman Data (Data Understanding)

### Sumber Data
- **Tautan:** [Bank Marketing Campaigns Dataset | Kaggle](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset/code)
- **Judul:** *Bank marketing campaigns dataset | Opening Deposit*

### Kamus Data (Attribute Information)

| Atribut          | Tipe Data | Jenis Data  | Deskripsi                                                                    |
| ---------------- | --------- | ----------- | ---------------------------------------------------------------------------- |
| `age`              | `int64`   | Numerik     | Usia klien                                                                   |
| `job`              | `object`  | Kategorikal | Jenis Pekerjaan Klien                                                        |
| `marital`          | `object`  | Kategorikal | Status Pernikahan Klien                                                      |
| `education`        | `object`  | Kategorikal | Tingkat pendidikan klien                                                     |
| `default`          | `object`  | Kategorikal | Status kredit klien                                                          |
| `housing`          | `object`  | Kategorikal | Status apakah klien memiliki pinjamanan perumahan                            |
| `loan`             | `object`  | Kategorikal | Status apakah klien memiliki pinjaman pribadi                                |
| `contact`          | `object`  | Kategorikal | Jenis komunikasi kontak terakhir                                             |
| `month`            | `object`  | Kategorikal | Bulan kontak terakhir dengan klien dalam tahun ini                           |
| `day_of_week`      | `object`  | Kategorikal | Hari dalam seminggu kontak terakhir dengan klien                             |
| `duration`         | `int64`   | Numerikal   | Durasi kontak terakhir dengan klien dalam detik                              |
| `campaign`         | `int64`   | Numerikal   | Jumlah kontak yang dilakukan selama kampanye ini dan untuk klien ini          |
| `pdays`            | `int64`   | Numerikal   | Jumlah hari sejak klien terakhir kali dihubungi dari kampanye sebelumnya     |
| `previous`         | `int64`   | Numerikal   | Jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini          |
| `poutcome`         | `object`  | Kategorikal | Hasil dari kampanye pemasaran sebelumnya                                     |
| `emp.var.rate`     | `float64` | Numerikal   | Tingkat variasi pekerjaan                                                    |
| `cons.price.idx`   | `float64` | Numerikal   | Indeks harga konsumen                                                        |
| `cons.conf.idx`    | `float64` | Numerikal   | Indeks kepercayaan konsumen                                                  |
| `euribor3m`        | `float64` | Numerikal   | Tingkat euribor 3 bulan                                                      |
| `nr.employed`      | `float64` | Numerikal   | Jumlah karyawan                                                              |
| `y` (Target)       | `object`  | Kategorikal | Status apakah klien telah berlangganan deposito berjangka? (`yes`/`no`)        |
| `call_fee`         | `float64` | Numerikal   | (Rekayasa Fitur) Biaya Penggunaan Telepon                                    |
| `duration_cut_off` | `int64`   | Numerikal   | (Rekayasa Fitur) Durasi kontak "cut off" dengan klien dalam detik            |
| `call_fee_cut_off` | `float64` | Numerikal   | (Rekayasa Fitur) Biaya Penggunaan "cut off" Telepon                          |


### Karakteristik Dataset
- **Dimensi Data:**
  - Jumlah Baris: **41.188**
  - Jumlah Kolom: **24**
- **Dataset Tidak Seimbang (Imbalanced):** Variabel target `y` sangat didominasi oleh label `"no"`. Ini memerlukan teknik penanganan khusus saat pemodelan, seperti *oversampling* atau *undersampling*, untuk mencegah model menjadi bias terhadap kelas mayoritas.
- **Dominan Fitur Kategorikal:** Banyak fitur penting bersifat kategorikal (nominal dan ordinal), yang memerlukan proses *encoding* yang sesuai (misalnya, *one-hot encoding* atau *ordinal encoding*).
- **Unit Observasi:** Setiap baris data merepresentasikan **satu interaksi telemarketing**, bukan satu nasabah unik. Ini berarti seorang nasabah dapat muncul lebih dari satu kali dalam dataset jika dihubungi beberapa kali.
- **Potensi Kebocoran Data (Data Leakage):** Fitur `duration` hanya diketahui **setelah** panggilan selesai. Menggunakannya sebagai prediktor akan menyebabkan kebocoran data, karena model akan belajar dari informasi yang tidak akan tersedia pada saat prediksi dibuat (sebelum panggilan). Oleh karena itu, `duration` dan fitur turunannya (`call_fee`, `duration_cut_off`, `call_fee_cut_off`) harus **dihapus** dari set data pelatihan.
- **Nilai "Unknown":** Beberapa kolom kategorikal mengandung nilai `"unknown"`. Ini dianggap sebagai nilai yang hilang (*missing values*) dan perlu ditangani.

---

## Preprocessing dan Analisis Data Eksploratif (EDA)

### Penanganan Data Duplikat
- Ditemukan **24 baris data duplikat** dalam dataset.
- Baris-baris ini dihapus untuk memastikan kualitas dan integritas data, menyisakan **41.176** baris unik.

### Penanganan Nilai Hilang ("Unknown")
Nilai `"unknown"` diidentifikasi dalam beberapa kolom dan diperlakukan sebagai *missing values*. Pendekatan imputasi yang logis dan berbasis konteks diterapkan:
- **`default`:** Kolom ini memiliki persentase *missing* tertinggi (**20.88%**). Mengingat mayoritas nasabah tidak memiliki kredit macet, nilai `"unknown"` diimputasi dengan nilai modus, yaitu `"no"`.
- **`housing` & `loan`:** Kedua kolom ini memiliki pola *missing* yang identik (**korelasi nulitas 1.0**), di mana jika salah satunya *missing*, yang lain juga *missing*. Analisis menunjukkan bahwa profil nasabah dengan nilai *missing* pada kedua kolom ini sangat mirip dengan profil nasabah yang tidak memiliki pinjaman sama sekali. Oleh karena itu, nilai `"unknown"` pada kedua kolom ini diimputasi dengan `"no"`.
- **`job` & `education`:** Kedua kolom ini memiliki hubungan. Imputasi dilakukan secara bertahap berdasarkan fitur lain yang berkorelasi:
  - Nasabah dengan `education` 'basic' cenderung memiliki pekerjaan 'blue-collar'.
  - Nasabah dengan `education` 'university.degree' atau 'high.school' cenderung memiliki pekerjaan 'admin.'.
  - Nasabah berusia > 60 tahun diimputasi dengan pekerjaan 'retired'.
  - Nasabah dengan `education` 'professional.course' cenderung menjadi 'technician'.
  - Sisa nilai *unknown* pada `education` diisi berdasarkan modus pekerjaan yang terkait.

### Transformasi Fitur
Untuk mempermudah analisis dan pemodelan, beberapa fitur ordinal diubah menjadi numerik:
- `month`: Diubah dari nama bulan (`"jan"`, `"feb"`, ...) menjadi angka (1-12).
- `day_of_week`: Diubah dari nama hari (`"mon"`, `"tue"`, ...) menjadi angka (1-5).

### Wawasan dari EDA
Analisis data eksploratif memberikan beberapa wawasan kunci mengenai profil nasabah yang cenderung membuka deposito:
- **Pekerjaan:** `student` dan `retired` memiliki tingkat konversi tertinggi, meskipun `admin.` memberikan kontribusi terbesar terhadap jumlah total deposito karena populasinya yang besar.
- **Status Pernikahan:** Nasabah `single` memiliki tingkat konversi yang sedikit lebih tinggi, namun nasabah `married` merupakan segmen terbesar.
- **Pendidikan:** Tingkat konversi tertinggi ditemukan pada nasabah dengan pendidikan `illiterate` (meskipun jumlahnya sangat kecil) dan `university.degree`. Ini menunjukkan bahwa tingkat pendidikan tinggi berkorelasi positif dengan kecenderungan membuka deposito.
- **Bulan Kontak:** Tingkat konversi tertinggi terjadi pada bulan **Maret, Desember, September, dan Oktober**. Sebaliknya, bulan **Mei** memiliki jumlah penawaran tertinggi tetapi tingkat konversi terendah, mungkin karena kampanye massal yang kurang tertarget.
- **Riwayat Kampanye (`poutcome`):** Nasabah yang **berhasil** dikonversi pada kampanye sebelumnya memiliki tingkat konversi yang sangat tinggi (**65%**) pada kampanye saat ini.

---

## Persiapan Data untuk Pemodelan

### Pencegahan Kebocoran Data (Data Leakage)
Seperti yang telah diidentifikasi, kolom `duration` dan turunannya akan menyebabkan kebocoran data. Oleh karena itu, kolom-kolom berikut dihapus dari dataset sebelum pemodelan:
- `duration`
- `call_fee`
- `duration_cut_off`
- `call_fee_cut_off`
- `month` dan `day_of_week` (versi string asli)
- `status_pinjaman` (kolom bantuan untuk EDA)

### Encoding Fitur Kategorikal
Fitur-fitur kategorikal yang tersisa diubah menjadi format numerik yang dapat diproses oleh model machine learning:
- **One-Hot Encoding:** Diterapkan pada fitur nominal yang tidak memiliki urutan intrinsik:
  - `job`, `marital`, `default`, `loan`, `housing`, `contact`, `poutcome`.
  - `drop='first'` digunakan untuk menghindari multikolinearitas.
- **Ordinal Encoding:** Diterapkan pada fitur `education` karena memiliki tingkatan yang jelas:
  - `illiterate` (0) < `basic.4y` (1) < ... < `university.degree` (6).

### Pemisahan Data (Train-Test Split)
Dataset dibagi menjadi data latih dan data uji dengan proporsi **80:20**. *Stratified splitting* digunakan untuk memastikan bahwa proporsi kelas `yes` dan `no` pada variabel target tetap sama di kedua set data, yang sangat penting untuk dataset yang tidak seimbang.

---

## Pemodelan dan Evaluasi

### Benchmarking Model Awal
Empat model klasifikasi populer dipilih untuk *benchmarking* awal tanpa penanganan ketidakseimbangan kelas. Evaluasi dilakukan menggunakan *5-fold cross-validation* dengan metrik **Recall**.
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Hasil *cross-validation* menunjukkan bahwa **Decision Tree** memiliki skor *recall* rata-rata tertinggi, meskipun model lain seperti Random Forest dan XGBoost menunjukkan *precision* yang lebih baik pada pengujian awal.

### Penanganan Ketidakseimbangan Kelas (Class Imbalance)
Karena skor *recall* awal masih rendah, teknik penanganan ketidakseimbangan kelas diterapkan. **RandomOverSampler** digunakan untuk menyeimbangkan distribusi kelas pada data latih dengan membuat duplikat dari sampel kelas minoritas (`yes`).

### Hyperparameter Tuning
Model **LightGBM** dipilih untuk optimasi lebih lanjut karena kinerjanya yang baik dan efisiensi komputasi. **RandomizedSearchCV** digunakan untuk mencari kombinasi *hyperparameter* terbaik dengan tujuan memaksimalkan skor **Recall**.

Parameter yang dioptimalkan antara lain:
- `n_estimators`: Jumlah pohon.
- `learning_rate`: Laju pembelajaran.
- `num_leaves`: Kompleksitas pohon.
- `min_child_samples`: Regularisasi untuk mencegah *overfitting*.
- `subsample` & `colsample_bytree`: Untuk meningkatkan kecepatan dan robustisitas.

Hasil *tuning* terbaik memberikan parameter berikut:
- `subsample`: 0.9
- `num_leaves`: 20
- `n_estimators`: 100
- `min_child_samples`: 20
- `learning_rate`: 0.1
- `colsample_bytree`: 0.9

Model yang telah dioptimalkan dengan *oversampling* dan *hyperparameter tuning* menunjukkan peningkatan signifikan pada **Recall** untuk kelas positif, dari sekitar 27% menjadi **64%** pada data validasi.

### Analisis Feature Importance
Analisis *feature importance* dari model LightGBM terbaik menunjukkan bahwa fitur-fitur berikut memiliki pengaruh terbesar dalam memprediksi apakah nasabah akan membuka deposito:
1.  **Indikator Ekonomi:** `euribor3m`, `emp.var.rate`, `nr.employed`, `cons.price.idx`, `cons.conf.idx` menunjukkan bahwa kondisi ekonomi makro saat kampanye berlangsung sangat berpengaruh.
2.  **Riwayat Kontak:** `pdays` (jumlah hari sejak kontak terakhir) dan `previous` (jumlah kontak sebelumnya) adalah prediktor kuat.
3.  **Hasil Kampanye Sebelumnya:** `poutcome_success` adalah fitur paling kuat, menandakan bahwa nasabah yang pernah berhasil dikonversi sangat berharga.
4.  **Profil Nasabah:** `age` dan `education` juga memberikan kontribusi yang signifikan.

---

## Kesimpulan dan Rekomendasi

### Ringkasan Hasil
Proyek ini berhasil mengembangkan model prediksi yang mampu mengidentifikasi nasabah potensial untuk penawaran deposito berjangka dengan **skor Recall sebesar 64%**. Ini berarti model mampu menangkap hampir dua pertiga dari semua nasabah yang sebenarnya tertarik, sebuah peningkatan signifikan yang dapat mengurangi kehilangan peluang bisnis.

Selain itu, analisis biaya menunjukkan bahwa penerapan strategi **batas waktu panggilan (cut-off)** dapat menghasilkan **penghematan biaya hingga 45%** dan **penghematan waktu agen hingga 43,68%**, membuktikan bahwa pendekatan berbasis data ini sangat efektif untuk meningkatkan efisiensi operasional.

### Rekomendasi Bisnis
1.  **Implementasi Model untuk Prioritas Panggilan:** Gunakan model prediksi untuk membuat daftar panggilan prioritas bagi tim telemarketing. Nasabah dengan probabilitas "yes" tertinggi harus dihubungi terlebih dahulu.
2.  **Penargetan Berjenjang (Tiered Targeting):** Kelompokkan nasabah ke dalam tingkatan prioritas (misalnya, *High, Medium, Low*) berdasarkan skor probabilitas dari model. Alokasikan lebih banyak sumber daya dan waktu untuk nasabah di tingkat *High*.
3.  **Memperkaya Dataset:** Untuk meningkatkan akurasi model di masa depan, pertimbangkan untuk menambahkan fitur-fitur baru seperti:
    - Data transaksional nasabah (misalnya, saldo rata-rata, jumlah produk yang dimiliki).
    - Data perilaku digital (misalnya, interaksi dengan situs web atau aplikasi bank).
    - Informasi pendapatan atau kekayaan nasabah.
4.  **Pemantauan dan Pelatihan Ulang Model:** Lakukan pemantauan performa model secara berkala dan latih ulang model (misalnya, setiap kuartal atau semester) dengan data baru untuk beradaptasi dengan perubahan tren pasar dan perilaku nasabah.

### Implementasi Model
Model ini dapat diimplementasikan sebagai berikut:
- Tim Data Scientist memberikan *dataset* baru (daftar nasabah yang akan dihubungi) ke model.
- Model akan menghasilkan skor probabilitas untuk setiap nasabah.
- Daftar nasabah kemudian diurutkan berdasarkan skor probabilitas ini dan diberikan kepada tim telemarketing.
- Tim telemarketing fokus pada nasabah dengan skor tertinggi, yang memungkinkan alokasi sumber daya yang lebih efektif dan meningkatkan tingkat keberhasilan kampanye secara keseluruhan.

### Limitasi Model
- **Ketidakseimbangan Kelas:** Meskipun ditangani dengan *oversampling*, sifat data yang tidak seimbang tetap menjadi tantangan. Model mungkin masih lebih baik dalam memprediksi kelas mayoritas (`"no"`).
- **Data Statis:** Model dilatih pada data dari periode waktu tertentu. Kinerjanya mungkin menurun seiring waktu karena perubahan kondisi ekonomi atau perilaku konsumen.
- **Trade-off Recall vs. Precision:** Fokus pada **Recall** akan meningkatkan jumlah *False Positives*, yang berarti tim marketing mungkin masih akan menghubungi beberapa nasabah yang tidak tertarik.
- **Generalisasi Geografis:** Model ini dilatih pada data dari Portugal. Kinerjanya mungkin tidak sama jika diterapkan di negara lain dengan kondisi pasar dan budaya yang berbeda.
