# Scientia Frontend

Antarmuka web untuk upload citra X-Ray dan menampilkan hasil prediksi pneumonia.

## Teknologi

- **HTML5** — struktur halaman
- **CSS3** — styling (dark theme, responsive)
- **JavaScript (Native/Vanilla)** — logika UI & komunikasi ke API

Tidak ada framework atau build tool — cukup buka langsung di browser.

## Struktur

```
frontend/
├── index.html      # Halaman utama
├── css/
│   └── style.css   # Semua styling
└── js/
    └── app.js      # Logika upload, fetch API, tampil hasil
```

## Menjalankan

```bash
# Dari folder frontend
python3 -m http.server 3000
```

Buka browser: `http://127.0.0.1:3000`

> Gunakan `127.0.0.1`, bukan `localhost`, agar konsisten dengan API di port 8000.

## Konfigurasi

URL API ada di baris pertama `js/app.js`:

```javascript
const API_BASE = "http://127.0.0.1:8000";
```

Ubah sesuai alamat server API jika deploy ke server lain.

## Fitur

- **Drag & drop** upload gambar atau klik "Pilih File"
- **Preview** gambar X-Ray sebelum dianalisis (grayscale)
- **Loading overlay** saat menunggu respons API
- **Hasil prediksi** — label (NORMAL / PNEUMONIA), confidence score, probability bar
- **Badge mode** — indikator apakah API berjalan di mode dummy atau real model
- **Error toast** — notifikasi jika koneksi ke API gagal

## Catatan

- Pastikan API sudah berjalan di port 8000 sebelum menggunakan fitur analisis
- Format gambar yang diterima: JPEG, PNG, maksimal 10 MB
- Hasil prediksi bukan pengganti diagnosis medis profesional
