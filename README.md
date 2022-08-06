# Klasifikasi-Citra-CT-Scan-Dada-Normal-dan-Covid-19-menggunakan-Machine-Learning
Dilakukan training pada model untuk memprediksi hasil klasifikasi citra CT-scan dada normal dan Covid-19 dengan convolutional neural network(CNN). 

## Dataset
Digunakan dataset yang mencangkup 746 Citra. Terdapat 349 citra positif Covid-19 dan 397 citra negatif Covid-19. Citra dibagi menjadi 522 citra untuk training, 
112 citra untuk validation, dan 112 citra untuk testing. Dataset dapat diakses [disini](https://github.com/UCSD-AI4H/COVID-CT).

## Metode
### Pre-processing
Setiap citra diproses sebelum memasuki neural network dengan langkah sebagai berikut :
1. Crop bagian citra yang memiliki gambar CT-scan dada.
2. Mengubah ukuran citra yang bervariasi masing-masing menjadi (240,240,3)=(panjang, lebar, jumlah channel warna)
3. Normalisasi setiap citra sehingga setiap pixel memiliki nilai 0-1.

### Feature learning
Pertama barisan dan kolom 0 (zeros) diletakan diatas, bawah, kanan dan kiri citra input dengan ZeroPadding2D. Hal ini akan memperluas jangkauan filter pada citra.
Kemudian dilakukan operasi konvolusi antara filters dan input image untuk mengekstraksi features dalam bentuk feature maps. Feature maps yang dihasilkan ditumpuk menjadi matriks
3 dimensi, dengan panjang dan lebar matriks sama dengan panjang dan lebar citra dan kedalaman matriks sama dengan jumlah feature map yang dihasilkan. lapisan BatchNormalization
kemudian menormalisasi nilai pixel untuk mempercepat komputasi. Non-linearitas diimplementasikan pada model dengan activation function ReLU (Rectifier linear units) sehingga weighted inputs yang negative dibuat 0. Kemudian dilakukan pooling dengan 
max pooling. Dengan mengambil setiap nilai maximum dari patch yang digeser di feature map, max pooling mengurangi dimensi feature map dan membuat representasi yang invariant
terhadap perubahan kecil pada gambar (membuat feature spasial lebih compact).  

### Classification
Hasil matrix 3 dimensi dari max pooling di flattened menjadi vektor 1 dimensi. Vektor ini kemudian menjadi input ke fully connected neural network dimana setiap neuron 
terkoneksi dengan neuron di layer berikutnya. Activation function yang digunakan adalah ReLu yang performanya lebih baik dan mempercepat training. Dengan weighted 
feature, model berusaha memprediksi hasil klasfikasi. Hasil prediksi kemudian dibandingkan dengan label training data. Model pada saat itu kemudian dijalankan pada 
validation data set. Error dari hasil prediksi training tersebut direpresentasikan dengan loss function (used function : Binary crossentropy). Untuk membuat hasil loss function sekecil mungkin, 
weights diubah dalam besaran tertentu(learning rate) agar hasil loss function menurun (used algorithm :Adam) dengan backpropogation. Learning rate ditentukan secara 
adaptif terhadap training data menggunakan algoritma optimisasi Adam, sehingga komputasi lebih efisien namun tetap efektif. Hal ini dilakukan secara iteratif hingga hasil loss function minimum.

### Architecture
Citra input dengan ukuran (240,240,3) dimasukan pada layer berikut :
1. ZeroPadding2D dengan pool size (2,2).
2. Konvolusi dengan 32 filter, ukuran filter =(7,7), langkah/stride = 1
3. BatchNormalization
4. Fungsi Aktifasi ReLu
5. Max pooling, ukuran=(4,4)
6. Flatten untuk membuat matriks 3 dimenso menjadi vektor 1 dimensi
7. fully connected neural network 
8. neuron dengan sigmoid activation function (ranges from 0-1, applicable for binary classification)

## Result
![image](https://user-images.githubusercontent.com/110709194/183257759-fc6b1c60-8687-4969-b6a8-18984bbf85fa.png)

Training loss menunjukan kinerja model dalam menghasilkan prediksi pada label training data dengan benar (loss function over time). 
Validation loss menunjukan kinerja model dalam menghasilkan prediksi pada label data baru/data yang tidak digunakan untuk training/validation data dengan benar (loss functoiion over time).
Training accuracy mengkomparasi hasil prediksi model dengan label data training dalam persentase.
Validation accuracy mengkomparasi hasil prediksi model dengan label data validasi dalam persentase.

Validation loss secara konsisten lebih rendah dari training loss, jarak diantaranya kurang lebih tetap sama sepanjang iterasi. Validation loss berfluktuasi.
