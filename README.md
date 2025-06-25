# Proyek Ujian Akhir Semester Statistika: Analisis Faktor-Faktor yang Mempengaruhi IPK

Ini adalah repositori untuk proyek Ujian Akhir Semester (UAS) mata kuliah Statistika, Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA), Universitas Negeri Semarang (UNNES) untuk semester Genap 2024/2025.

**Dosen Pengampu:** Zaenal Abidin, S.Si., M.Sc., Ph.D

---

## 📝 Deskripsi Proyek

Proyek ini bertujuan untuk menerapkan metode statistika dalam menganalisis faktor-faktor yang berpotensi memengaruhi prestasi akademik mahasiswa, yang diukur melalui Indeks Prestasi Kumulatif (IPK). Kasus yang diangkat adalah untuk menguji hipotesis apakah **Jam Belajar per Minggu** dan **Tingkat Kehadiran** memiliki pengaruh yang signifikan terhadap IPK mahasiswa.

Analisis dilakukan menggunakan bahasa pemrograman Python dengan bantuan pustaka ilmiah populer.

## 🛠️ Metodologi Analisis

Analisis data dalam proyek ini mencakup beberapa tahapan utama:

1.  **Statistik Deskriptif**: Untuk memberikan gambaran umum data.
2.  **Uji Korelasi Pearson**: Untuk mengukur kekuatan hubungan linear antar variabel.
3.  **Analisis Regresi Linear Berganda**: Untuk memodelkan pengaruh simultan variabel independen terhadap variabel dependen.
4.  **Uji Asumsi Klasik**: Meliputi uji normalitas, multikolinearitas, heteroskedastisitas, dan autokorelasi untuk memastikan validitas model regresi.

## 📈 Hasil Utama

Berdasarkan analisis terhadap 150 sampel data, ditemukan bahwa **tidak terdapat pengaruh yang signifikan secara statistik** dari variabel Jam Belajar per Minggu dan Tingkat Kehadiran terhadap IPK mahasiswa. Nilai R-squared yang sangat rendah (0.001) menunjukkan bahwa kedua variabel tersebut bukan merupakan prediktor yang baik untuk prestasi akademik dalam konteks data ini.

## 🚀 Cara Menjalankan Kode

Untuk mereplikasi hasil analisis, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**

    ```bash
    git clone [tempel_link_repo_anda_di_sini]
    cd [nama_folder_repo_anda]
    ```

2.  **Buat dan aktifkan virtual environment (opsional, tapi direkomendasikan):**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal semua pustaka yang dibutuhkan:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan skrip analisis:**
    ```bash
    python main.py
    ```
    Skrip akan mencetak semua hasil tabel analisis di terminal dan menyimpan plot visualisasi di dalam folder `/visualisasi/`.

## 🧑‍🎓 Informasi Mahasiswa

- **Nama:** Muhamad Nur Iman
- **NIM:** 2404140048
- **Program Studi:** SISTEM INFORMASI'24 ROMBEL 1
