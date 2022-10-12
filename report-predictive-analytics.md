# Laporan Proyek Machine Learning - Moch Galang Rivaldo

## Domain Proyek

Pembangkit listrik gabungan tenaga dua turbin gas, satu turbin uap dan dua sistem pemulihan panas menghasilkan parameter-parameter (tekanan udara, suhu, dan lainnya) yang berhubungan dengan besarnya keluaran daya listrik yang dihasilkan. Efisiensi dari segi operasional perlu ditingkatkan untuk meningkatkan ekonomi atau pendapatan dari pembangkit listrik tersebut. Hal ini berguna untuk memaksimalkan pendapatan dari daya listrik yang dihasilkan (Megawatt Hours / MWh). Perusahaan pembangkit listrik tersebut sangat bergantung pada prediksi daya yang dihasilkan karena adanya perjanjian profitabilitas yang tinggi dan kontrak lainnya.

Pınar Tüfekci (2014) melakukan prediksi keluaran daya listrik pada perusahaan pembangkit listrik tersebut dengan metode regresi (model REPTree) dan teknik *Bagging* (metode *ensemble*) diperoleh nilai metrik *Mean Absolute Error* 2.818 dan *Root Mean-Squared Error* 3.787 [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0142061514000908?via%3Dihub). Sejalan dengan hasil penelitian tersebut, solusi yang ditawarkan adalah dengan memanfaatkan *machine learning* dengan metode regresi untuk memprediksi keluaran daya listrik yang berguna menjadi pedoman dalam meningkatkan efisiensi operasional yang berdampak pada peningkatan pendapatan.

## Business Understanding


### Problem Statements

- Perusahaan pembangkit listrik membutuhkan model terbaik untuk melakukan prediksi keluaran daya listrik yang berguna menjadi pedoman dalam meningkatkan efisiensi operasional yang berdampak pada peningkatan pendapatan.

### Goals

- Membangun model terbaik untuk melakukan prediksi keluaran daya listrik yang berguna menjadi pedoman dalam meningkatkan efisiensi operasional yang berdampak pada peningkatan pendapatan.

### Solution Statements
- Menawarkan solusi sistem prediksi dengan metode regresi. Untuk mendapatkan solusi terbaik, akan digunakan tiga model yang berbeda (KNN, RandomForest, Boosting) dengan *hyperparameter tuning*. Selain itu, untuk mengukur kinerja model digunakan metrik *Mean Squared Error* (MSE) dimana model terbaik nantinya harus memperoleh nilai MSE terkecil dari dataset uji.

## Data Understanding

Berdasarkan sumber dataset: [UCI Machine Learning Repository - Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) diperoleh informasi:  
**Abstrak**: Dataset terdiri dari 9568 data point (baris) yang dikumpulkan dari Pembangkit Tenaga Listrik selama 6 tahun (2006 - 2011)

Tabel 1. Informasi Dataset

|       | Description |
| ----------- | ----------- |
| Data Set Characteristics | Multivariate |
| Attribute Characteristics | Real |
| Associated Tasks | Regression |
| Number of Instances | 9568 |
| Number of Attributes | 4 |
| Missing Values? | N/A |
| Area | Computer |

### Variabel-variabel pada *Combined Cycle Power Plant* Dataset adalah sebagai berikut:
* *Temperature* (T) adalah suhu dengan jangkauan 1,81 sampai 37,11 (dalam derajat *Celcius*)
* *Ambient Pressure* (AP) adalah satuan tekanan dari media di sekitar objek seperti gas atau cairan dengan jangkauan 992,89 sampai 1033,30 milibar.
* *Relative Humidity* (RH) adalah jumlah kadar air (uap air) yang menandakan kemampuan udara untuk menampung uap air dengan jangkauan 25,56 sampai 100,16 (dalam persen %).
* *Exhaust Vacuum* (V) adalah kekuatan dalam menghisap debu maupun membuang udara panas dengan jangkauan 25,36 sampai 81,56 (dalam satuan cmHg).
* *Net hourly electrical energy output* (EP) adalah besarnya daya yang dihasilkan setiap satu jamnya dengan jangkauan 420,26 sampai 495, 76 (dalam satuan MW per jam).

### Menangani Missing Value
Untuk mendeteksi *missing value* digunakan fungsi `isnull().sum()` dan diperoleh:  
Tabel 2. Hasil Deteksi *Missing Value*
|Fitur | Jumlah *Missing Value*|
|:---:|:---:|
|AT | 0|
|V | 0|
|AP  |  0|
|RH  |  0|
|PE  |  0|

Dari Tabel 2. terlihat bahwa setiap fitur tidak memiliki *Missing Value* (NULL maupun NAN) sehingga dapat dilanjutkan ke tahapan selanjutnya yaitu menangani *outliers*.
### Menangani Outliers
Pada kasus ini, untuk mendeteksi *outliers* digunakan teknis visualisasi data (boxplot). Kemudian untuk menangani *outliers* digunakan metode IQR.  

Seltman dalam “Experimental Design and Analysis” [[2]](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf) menyatakan bahwa outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot outliers”) didefinisikan sebagai data yang nilainya 1.5 IQR di atas Q3 atau 1.5 IQR di bawah Q1.

Berikut persamaannya:
```
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```
Tabel 3. Visualisasi Boxplot Sebelum dan Sesudah Dikenakan Metode IQR.

| Cek Outlier Pada Fitur | Setelah Dikenakan Metode IQR |
|:---:|:---:|
| Fitur AT (Before)![](https://aneechan.github.io/assets/picture/fitur-AT-cek-outlier.png) | Fitur AT (After)![](https://aneechan.github.io/assets/picture/fitur-AT-iqr.png) |
| Fitur V (Before)![](https://aneechan.github.io/assets/picture/fitur-V-cek-outlier.png) | Fitur V (After)![](https://aneechan.github.io/assets/picture/fitur-V-iqr.png) |
| Fitur AP (Before)![](https://aneechan.github.io/assets/picture/fitur-AP-cek-outlier.png) | Fitur AP (After)![](https://aneechan.github.io/assets/picture/fitur-AP-iqr.png) |
| Fitur RH (Before)![](https://aneechan.github.io/assets/picture/fitur-RH-cek-outlier.png) | Fitur RH (After)![](https://aneechan.github.io/assets/picture/fitur-RH-iqr.png) |
| Fitur PE (Before)![](https://aneechan.github.io/assets/picture/fitur-PE-cek-outlier.png) | Fitur PE (After)![](https://aneechan.github.io/assets/picture/fitur-PE-iqr.png) |

Dari hasil deteksi ulang *outlier* dengan boxplot di Tabel 3 di atas, didapat bahwa *outlier* sudah berkurang pada tiap fitur setelah dibersihkan seperti pada Tabel 4. berikut.

Tabel 4. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outlier

| Jumlah Data Sebelum Dibersihkan | Jumlah Data Setelah Dibersihkan |
|:---:|:---:|
| 9568 | 9468 |

### Univariate Analysis
Selanjutnya, akan dilakukan proses analisis data dengan teknik Univariate EDA. Pada kasus ini semua fiturnya adalah fitur numerik dan tidak ada fitur kategorikal. Sehingga hanya perlu dilakukan analisa terhadap fitur numerik, sebagai berikut:
#### Analisa Fitur Numerik
Untuk melihat distribusi data pada tiap fitur akan digunakan visualisasi dengan histogram sebagai berikut:

![](https://aneechan.github.io/assets/picture/histogram-univariate-analysis.png)

Gambar 1. Histogram Pada Setiap Fitur

Berdasarkan Gambar 1. di atas, diperoleh beberapa informasi, antara lain:
* Distribusi fitur PE (target) cenderung miring ke kanan (*right-skewed*).
* Distibusi fitur RH cenderung miring ke kiri (*left-skewed*).

Karena beberapa fitur belum terdistribusi normal hal ini akan berimplikasi pada model, maka untuk selanjutnya perlu dilakukan transformasi data (*non-linear scaling*). Namun, sebelum itu akan dicek terlebih dahulu hubungan antara fitur numerik tersebut.

### Multivariate Analysis

#### Hubungan antara Fitur Numerik
Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi `pairplot()`, dengan *output* sebagai berikut:

![](https://aneechan.github.io/assets/picture/pairplot-multivariate-analysis.png)

Gambar 2. Visualisasi Hubungan antara Fitur Numerik dengan pairplot()

Pola sebaran data grafik pairplot pada Gambar 2. terlihat fitur `AT` dan `V` memiliki korelasi kuat (negatif / berkebalikan) dengan fitur `PE` (target). Sedangkan kedua fitur lainnya yaitu `AP` dan `RH` memiliki korelasi positif yang lemah dengan fitur `PE`.

#### Korelasi antara Fitur Numerik
Untuk mengevaluasi skor korelasi hubungan antara fitur numerik, akan digunakan fungsi `corr()` dengan *output* sebagai berikut.

![](https://aneechan.github.io/assets/picture/korelasi-multivariate-analysis.png)

Gambar 3. Korelasi antara Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Dari Gambar 3. di atas, fitur `AT` dan `V` memiliki korelasi yang kuat (mendekati -1, dibawah -0.85) dengan fitur target `PE`. Sementara itu, fitur `AP` dan `RH` mempunyai korelasi yang rendah dengan fitur target `PE`.

## Data Preparation

### Train Test Split
Dataset akan dibagi menjadi data latih (*train*) dan data uji (*test*). Tujuan langkah ini sebelum proses lainnya adalah agar tidak mengotori data uji dengan informasi yang didapat dari data latih. Contoh pada proses standarisasi dimana jika belum di bagi menjadi data latih dan uji, maka keduanya akan terkena transformasi data yang menggunakan informasi (*mean* dan *standard deviation*) dari gabungan data latih dan uji. Hal ini berpotensi menimbulkan kebocoran data (*data leakage*). Oleh karena itu langkah awal sebelum melakukan tranformasi data adalah membagi dataset terlebih dahulu [[3]](https://learning.oreilly.com/library/view/hands-on-predictive-analytics/9781789138719/).

Pada kasus ini akan menggunakan proporsi pembagian sebesar 90:10 dengan fungsi `train_test_split` dari sklearn dengan *output* sebagai berikut.

Tabel 5. Jumlah Data Latih dan Uji

| Jumlah Data Latih | Jumlah Data Uji | Jumlah Total Data |
|:---:|:---:|:---:|
| 8521 | 947 | 9468 |

### Standarisasi
Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada kasus ini akan digunakan metode `StandarScaler()` dari *library* Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan *mean* kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan *mean* sama dengan 0.

Berikut *output* yang dihasilkan dari metode StandardScaler dengan menggunakan fungsi `describe()`:

Tabel 6. Hasil Proses Standarisasi Pada Setiap Fitur Pada Data Latih

| |AT | V | AP | RH |
|:---:|---:|---:|---:|---:|
| count | 8521.0000 | 8521.0000 | 8521.0000| 8521.0000 |
| mean |-0.0000 | -0.0000 | 0.0000 | -0.0000 |
| std | 1.0001 | 1.0001 | 1.0001 | 1.0001 |
| min | -2.4059 | -2.2844 | -2.8439 | -2.9015 |
| 25% | -0.8254 | -0.9953 | -0.7095 | -0.6870 |
| 50% | 0.0973 | -0.1595 | -0.0423 | 0.1132 |
| 75% | 0.8129 | 0.9565 | 0.7000 | 0.7904 |
| max | 2.3422 | 2.1385 | 2.8396 | 1.8386 |

### Non-Linear Scaling
Terkait hasil visualisasi histogram setiap fitur sebelumnya, terdapat beberapa fitur yang belum terdistribusi normal, antara lain fitur `PE` (*right-skewed*) dan fitur `RH` (*left-skewed*). Sehingga dapat dilakukan proses *non-linear scaling*. Pada kasus ini, proses *non-linear scaling* akan menggunakan metode `Yeo-Johnson` karena dapat menangani data negatif setelah proses standarisasi sebelumnya.

Mengingat fitur `PE` adalah target fitur, maka akan dikecualikan dalam proses ini agar distribusinya tetap dipertahankan sesuai data aslinya.

Berikut visualisasi dengan histogram untuk distribusi data pada setiap fitur setelah dilakukan *non-linear scaling* dengan metode `Yeo-Johnson`.

![](https://aneechan.github.io/assets/picture/non-linear-scaling-at.png)

Gambar 4. Histogram Fitur AT Sebelum (Kiri) dan Sesudah (Kanan) Dikenakan Metode Yeo-Johnson

![](https://aneechan.github.io/assets/picture/non-linear-scaling-v.png)

Gambar 5. Histogram Fitur V Sebelum (Kiri) dan Sesudah (Kanan) Dikenakan Metode Yeo-Johnson

![](https://aneechan.github.io/assets/picture/non-linear-scaling.png)

Gambar 6. Histogram Fitur AP Sebelum (Kiri) dan Sesudah (Kanan) Dikenakan Metode Yeo-Johnson

![](https://aneechan.github.io/assets/picture/non-linear-scaling-rh.png)

Gambar 7. Histogram Fitur RH Sebelum (Kiri) dan Sesudah (Kanan) Dikenakan Metode Yeo-Johnson

Selanjutnya dilakukan *drop* pada fitur AT, V, AP, RH karena sudah tergantikan dengan YJ_AT, YJ_V, YJ_AP, YJ_RH yang lebih mendekati distribusi normal. Berikut contoh 5 teratas dari data latih dengan fungsi `head()` untuk mengecek hasil *drop* fitur.

Tabel 7. Tampilan 5 Teratas dari Data Latih Setelah Dilakukan *Drop*

| | YJ_AT | YJ_V | YJ_AP | YJ_RH |
|:---:|---:|---:|---:|---:|
| 0 | -0.935750 | -0.130613 | 2.089722 | -0.307734 |
| 1 | 1.336847 | 0.760638 | 0.754588 | -1.647642 |
| 2	| 0.727460 | 0.724802 | 0.064009 | -0.556362 |
| 3	| -0.635213 | -0.660453 | -1.036064 | 1.512621 |
| 4	| 0.327177 | -1.145919 | -2.048456 | -1.529602 |

### Reduksi Dimensi dengan PCA
PCA umumnya digunakan ketika variabel dalam data yang memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau *redundant*. Sebelumnya sudah dilakukan proses untuk melihat hubungan dan korelasi dengan `pairplot()`, namun setelah melewati proses transformasi data (standarisasi dan *non-linear scaling*) dimungkinkan terjadi perubahan korelasi antar fiturnya (meskipun relatif kecil). Untuk itu perlu dicek kembali korelasi antar fitur dengan menggunakan `pairplot()` dengan *output* sebagai berikut.

![](https://aneechan.github.io/assets/picture/pairplot-multivariate-analysis-yj.png)

Gambar 8. Visualisasi Hubungan antara Fitur Numerik dengan pairplot() pada Data Latih

Selanjutnya perlu dilakukan reduksi fitur `YJ_AT` dan `YJ_V` karena keduanya berkorelasi kuat yang dapat dilihat pada Gambar 8. di atas.

Namun, sebelum perlu dilihat proporsi informasi dari komponen-komponen (*Principal Component*) hasil dari PCA yang nantinya dijadikan patokan dalam mereduksi fitur. Untuk implementasinya menggunakan fungsi `PCA()` dari sklearn dengan mengatur nilai parameter `n_components` sebanyak fitur yang akan dikenakan PCA dalam hal ini ada dua yaitu YJ_AT dan YJ_V.

Tabel 8. Proporsi *Principal Component* Dari Hasil PCA Fitur YJ_AT dan YJ_V

| PC Pertama | PC Kedua |
|:---:|:---:|
|0.925|0.075|

Arti dari Tabel 8. di atas adalah, 92.5% informasi pada kedua fitur `YJ_AT` dan `YJ_V` terdapat pada PC (*Principal Component*) pertama. Sedangkan sisanya sebesar 7.5% terdapat pada PC kedua.

Berdasarkan hasil tersebut, selanjutnya akan dilakukan reduksi fitur dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur baru yang menggantikan dua fitur lainnya (`YJ_AT` dan `YJ_V`). Fitur baru tersebut diberi nama `PC_ATV` (*Principal Component* AT & V). Untuk implementasinya diubah nilai parameter `n_components` menjadi 1 (PC Pertama).

Berikut tampilan data latih dengan fungsi `sample(5)` setelah dilakukan reduksi fitur.

Tabel 9. Tampilan 5 Sampel dari Data Latih Setelah Dilakukan Reduksi Fitur YJ_AT dan YJ_V menjadi Fitur PC_ATV
| | YJ_AP | YJ_RH | PC_ATV |
|:---:|---:|---:|---:|
| 6697 | 0.489467 | 0.281716 | -0.668645 |
| 3735 | 2.049647 | 1.627538 | -1.298155 |
| 52 | 0.110746 | 1.318874 | 0.339899 |
| 5419 | -0.061729 | 1.538340 | 0.730265 |
| 5523 | 0.158992 | -0.439972 | 1.271907 |

### Transformasi Data Uji
Sebelumnya, telah dilakukan proses transformasi data (standarisasi dan *non-linear scaling*) pada data latih untuk menghindari kebocoran data. Sekarang, setelah data latih ditransformasikan secara independen dan diamankan ke dalam variable `X_train_scaled`, selanjutnya perlu melakukan proses transformasi data terhadap data uji dengan `scaler` dari proses standarisasi, `yj_scaler` dari proses *non-linear scaling* (metode `yeo-johnson`) dan proses pca untuk digunakan pada proses evaluasi model.

Biasanya proses ini dilakukan setelah proses *training* model, namun akan dilakukan sekarang dengan tujuan supaya dapat digunakan untuk mencari nilai k optimum pada model KNN (bagian selanjutnya).

Tabel 10. Tampilan Data Uji Setelah Proses Transformasi Data
| | YJ_AP | YJ_RH | PC_ATV |
|:---:|---:|---:|---:|
| 0 | -0.280138 | 1.048203 | 0.682992 |
| 1 | -0.044167 | 0.784226 | -1.763109 |
| 2 | -1.100085 | -0.907001 | 0.864835 |
| 3 | 1.495946 | 0.510210 | -1.958569 |
| 4 | 0.398838 | 0.552927 | -1.802959 |
| ... | ... | ... | ... |
| 942 | -0.362872 | -0.638957 | 1.633829 |
| 943 | 0.041428 | 0.444081 | 0.903385 |
| 944 | 0.937499 | -0.295727 | -1.636853 |
| 945 | -0.112832 | -1.697139 | 2.267231 |
| 946 | -1.034184 | -0.435081 | 0.333504 |

## Model Development
Pada tahap ini, akan menggunakan tiga algoritma untuk regresi. Kemudian, akan dilakukan evaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:
1. K-Nearest Neighbor

    Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya jika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest

    Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan *overfitting* dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree. 

3. Boosting Algorithm

    Kelebihan algoritma Boosting adalah menggunakan teknik Boosting yang berusaha menurunkan bias dengan berjalan secara sekuensial (memperbaiki model di tiap tahapnya). Sedangkan kekurangannya hampir sama dengan algoritma Random Forest dari segi kompleksitas komputasi yang menjadikan waktu pelatihan relatif lebih lama, selain itu *noisy* dan *outliers* sangat berpengaruh dalam algoritma ini.

Langkah pertama membuat DataFrame baru `df_models` untuk menampung nilai metrik pada setiap model / algoritma. Hal ini berguna untuk melakukan analisa perbandingan antar model. Metrik yang digunakan untuk mengevaluasi model adalah (MSE - *Mean Squared Error*).

### Model K-Nearest Neighbor
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang *overfitting* dan hasil prediksinya memiliki varians tinggi. Sedangkan jika memilih k yang terlalu tinggi, maka model yang dihasilkan akan *underfitting* dan prediksinya memiliki bias yang tinggi [[4]](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).

Oleh karena itu, perlu mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini memakai *mean squared error*) terbaik. Selain itu, akan digunakan metrik ukuran jarak secara default (*Minkowski Distance*) pada `KNeighborsRegressor` dari library sklearn.

Tabel 11. Perbandingan Nilai K terhadap Nilai MSE
| K | MSE |
|:---:|:---|
| 1 | 27.164278458289346 |
| 2 | 21.131600369588185 |
| 3 | 19.491511862020435 |
| 4 | 19.261315192713845 |
| 5 | 18.72220005068637 |
| 6 | 18.880029869177513 |
| 7 | 18.529073128892524 |
| 8 | 18.68825011054647 |
| 9 | 18.636591223747512 |
| 10 | 18.852110728616697 |
| 11 | 18.962034640055155 |
| 12 | 19.011120737269756 |
| 13 | 19.172104390070178 |
| 14 | 19.2599176567248 |
| 15 | 19.383454522585954 |
| 16 | 19.384363042832625 |
| 17 | 19.4375758187392 |
| 18 | 19.6279974950135 |
| 19 | 19.7668103885429 |
| 20 | 19.74753467951425 |

Jika divisualisasikan dengan fungsi `plot()` diperoleh:

![](https://aneechan.github.io/assets/picture/visualisasi-nilai-k-terhadap-mse.png)

Gambar 9. Visualisasi Nilai K terhadap MSE

Berdasarkan Tabel 11. dan Gambar 9. di atas, nilai MSE terbaik dicapai ketika `k = 7` yaitu sebesar 18.529 pada data uji, sehingga dipilih k = 7 dan menyimpan nilai MSE nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam DataFrame `df_models` yang telah disiapkan sebelumnya.

### Random Forest
Random forest merupakan algoritma *supervised learning* yang termasuk ke dalam kategori *ensemble* (group) learning. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. Jenis metode *ensemble* yang digunakan pada Random Forest adalah teknik *Bagging*. Metode ini bekerja dengan membuat *subset* dari data train yang independen. Beberapa model awal (*base model / weak model*) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data *train* yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final.

Untuk implementasinya menggunakan `RandomForestRegressor` dari *library* scikit-learn dengan `base_estimator` defaultnya yaitu `DecisionTreeRegressor` dan parameter-parameter (*hyperparameter*) yang digunakan antara lain:
- `n_estimator`: jumlah *trees* (pohon) di *forest*.
- `max_depth`:  kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (*splitting*) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- `random_state`: digunakan untuk mengontrol random *number generator* yang digunakan.
- `n_jobs`: jumlah *job* (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol *thread* atau proses yang berjalan secara paralel. `n_jobs`=-1 artinya semua proses berjalan secara paralel.

Untuk menentukan nilai *hyperparameter* (`n_estimator` & `max_depth`) di atas, dilakukan *tuning* dengan `RandomizedSearchCV` (5 *folds* untuk setiap 10 kandidat sehingga total 50 proses *fitting*) dan hasilnya sebagai berikut:

Tabel 12. Hasil *Hyperparameter Tuning* model RandomizedSearchCV dengan Random Forest
|  | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 80 |
| max_depth | 4, 8, 16, 32 | 32 |
| MSE Data Latih | | 2.592137498639827 |
| MSE Data Uji | | 17.254006379841524 |


Berdasarkan Tabel 12. di atas diperoleh nilai MSE terbaik dalam jangkauan parameter (daftar nilai) yaitu 2.592 (dengan data *train*) dan 17.254 (dengan data *test*) dengan `n_estimators`: 80 dan `max_depth`: 32. Selanjutnya dipilih pengaturan parameter tersebut dan menyimpan nilai MSE nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam `df_models` yang telah disiapkan sebelumnya.

### Boosting Algorithm
Jika sebelumnya telah digunakan algoritma *bagging* (Random Forest). Selanjutnya akan menggunakan metode lain dalam model *ensemble* yaitu teknik Boosting. Algoritma Boosting bekerja dengan membangun model dari data *train*. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Teknik ini bekerja secara sekuensial.

Pada kasus ini akan menggunakan metode Adaptive Boosting. Untuk implementasinya menggunakan `AdaBoostRegressor` dari library sklearn dengan `base_estimator` defaultnya yaitu `DecisionTreeRegressor` hampir sama dengan `RandomForestRegressor` bedanya menggunakan metode teknik Boosting.

Parameter-parameter (*hyperparameter*) yang digunakan pada algoritma ini antara lain:
* `n_estimator`: jumlah *estimator* dan ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
* `learning_rate`: bobot yang diterapkan pada setiap *regressor* di masing-masing iterasi Boosting.
* `random_state`: digunakan untuk mengontrol *random number generator* yang digunakan.

Untuk menentukan nilai *hyperparameter* (`n_estimator` & `learning_rate`) di atas, akan dilakukan *tuning* dengan `RandomizedSearchCV` (5 *folds* untuk setiap 10 kandidat sehingga total 50 proses *fitting*) dan hasilnya sebagai berikut:

Tabel 13. Hasil *Hyperparameter Tuning* model RandomizedSearchCV dengan AdaBoosting
|  | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 20 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.2 |
| MSE Data Latih | | 25.319184456840937 |
| MSE Data Uji | | 24.775647647800028 |

Berdasarkan Tabel 13. di atas diperoleh nilai MSE terbaik dalam jangkauan parameter (daftar nilai) yaitu 25.319 (dengan data *train*) dan 24.775 (dengan data *test*) dengan `n_estimators`: 20 dan `learning_rate`: 0.2. Selanjutnya dipilih pengaturan parameter tersebut dan menyimpan nilai MSE nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam `df_models` yang telah disiapkan sebelumnya.

### Model Terbaik berdasarkan Nilai MSE pada Data Latih
Pada tahap ini, hanya dibatasi pada data latih karena penggunaan data uji akan dilakukan pada proses evaluasi model. Berdasarkan DataFrame `df_models` diperoleh:

Tabel 14. Nilai MSE pada Setiap Model dengan Data Latih
| | KNN | RandomForest | Boosting |
|:---:|---:|---:|---:|
| Train MSE | 14.032666 | 2.61751 | 25.552998 |

Dari Tabel 14. di atas, perlu diperhatikan bahwa hasil MSE pada tabel sedikit berbeda dengan MSE hasil analisa proses *hyperparameter tuning* sebelumnya (khususnya pada Random Forest dan Boosting). Hal ini disebabkan model pada proses *hyperparameter tuning* menggunakan model *RandomizedSearchCV* berbeda dengan Tabel 14. yang menggunakan model Random Forest dan Boosting. Terlepas dari hal tersebut, model terbaik dipegang oleh Random Forest dengan nilai MSE 2.61751 (terkecil).

## Evaluation
Dari proses sebelumnya, telah dibangun dan dilatih tiga model yang berbeda (KNN, Random Forest, Boosting). Selanjutnya perlu mengevaluasi model-model tersebut menggunakan data uji dan metrik yang digunakan dalam kasus ini yaitu `mean_squared_error`. Hasil evaluasi kemudian disimpan ke dalam `df_models`.

![](https://www.gstatic.com/education/formulas2/472522532/en/mean_squared_error.svg)

Gambar 10. Formula MSE

Keterangan formula MSE pada gambar 10:
- MSE = *Mean Squared Error*
- n = banyaknya data point (baris)
- Y_i = nilai yang diobservasi (fitur target `PE`)
- Y^_i = hasil prediksi

Cara kerja metrik MSE adalah dengan menghitung selisih hasil prediksi dengan nilai fitur target (`PE`). Nilai selisih tersebut, disebut juga sebagai nilai eror yang kemudian di kuadratkan untuk menangani nilai selisih negatif, selanjutnya hasil pengkuadratan setiap nilai selisih dijumlahkan dan terakhir dibagi dengan banyak data point (n) untuk memperoleh nilai rata-ratanya. Rata-rata inilah yang disebut *Mean Squared Error* (MSE). Metrik MSE kerap digunakan untuk mengevaluasi model regresi seperti pada kasus ini.

Berdasarkan DataFrame `df_models` diperoleh:

Tabel 15. Nilai MSE pada Setiap Model dengan Data Uji

| | KNN | RandomForest | Boosting |
|:---:|---:|---:|---:|
| Test MSE | 18.529073 | 17.757302	 | 25.28377 |

Untuk memudahkan, dilakukan plot hasil evaluasi model dengan *bar chart* sebagai berikut:

![](https://aneechan.github.io/assets/picture/plot-hasil-evaluasi-model.png)

Gambar 11. Bar Chart Hasil Evaluasi Model dengan Data Latih dan Uji

Dari Gambar 10 dan 11 di atas, terlihat bahwa, model RandomForest memberikan nilai eror (MSE) yang paling kecil. Sedangkan model algoritma Boosting memiliki eror yang paling besar. Sebelum memutuskan model terbaik untuk melakukan prediksi *Net hourly electrical energy output* (EP) atau besarnya daya yang dihasilkan setiap satu jam. Perlu dilakukan uji prediksi menggunakan beberapa sampel acak (5) pada data uji dengan hasil sebagai berikut:

Tabel 16. Hasil Prediksi dari 5 Sampel Acak

| index_sample | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|:---:|:---:|:---:|:---:|:---:|
| 778 | 448.71 | 444.512857 | 445.660375 | 445.046823 |
| 844 | 443.91 | 440.494286 | 443.087750 | 441.048579 |
| 41 | 479.48 | 476.290000 | 478.694125 | 475.847589 |
| 162 | 435.53 | 434.881429 | 435.580250 | 440.357165 |
| 863 | 434.57 | 438.380000 | 436.956875 | 437.679482 |

Dari Tabel 16. terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati. Untuk penentuan model terbaik akan disampaikan di bagian kesimpulan.

## Conclusion

Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan prediksi *Net hourly electrical energy output* (EP) atau besarnya daya yang dihasilkan setiap satu jam adalah model Random Forest. Dengan pengaturan parameter `n_estimators`: 80, `max_depth`: 32 diperoleh nilai metrik MSE sebesar 2.617 (pada data latih) dan 17.757 (pada data uji). Diharapkan dengan dibangunnya model ini dapat menjadi pedoman perusahaan dalam mengefisiensikan operasional yang berdampak pada kenaikan pendapatan.

## Daftar Referensi
[1] Pınar Tüfekci, Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods, International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615, [[Web Link](http://dx.doi.org/10.1016/j.ijepes.2014.02.027)]. ([[Web Link](http://www.sciencedirect.com/science/article/pii/S0142061514000908)])  
[2] Seltman, Howard J. “Experimental Design and Analysis”. 2018. Tersedia: [tautan](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf). Diakses pada Oktober 2022.  
[3] Fuentes, Alvaro. "Hands-on Predictive Analytics with Python". Packt Publishing. 2018. Page 129. Tersedia: [O'Reilly Media](https://learning.oreilly.com/library/view/hands-on-predictive-analytics/9781789138719/).  
[4] Rhys, Hefin. "Machine Learning with R, the Tidyverse, and MLR". Manning Publications. 2020. Page 286. Tersedia: [O'Reilly Media](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).
