# Scientia API

REST API untuk deteksi pneumonia dari citra X-Ray dada menggunakan CNN binary classification.

## Teknologi

- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **PyTorch + TorchVision** — inference model CNN
- **Pillow** — preprocessing gambar

## Struktur

```
api/
├── app.py            # Entry point FastAPI
├── requirements.txt  # Daftar dependencies
└── models/
    └── pneumonia_cnn.pth   # File model (taruh di sini setelah training)
```

## Setup & Menjalankan

```bash
# 1. Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install fastapi "uvicorn[standard]" python-multipart pillow

# Setelah model tersedia, tambahkan:
pip install torch torchvision

# 3. Jalankan server
uvicorn app:app --reload --port 8000 --host 127.0.0.1
```

Server berjalan di `http://127.0.0.1:8000`

## Endpoints

| Method | Endpoint   | Deskripsi                        |
|--------|------------|----------------------------------|
| GET    | `/`        | Status API                       |
| GET    | `/health`  | Cek status server & model        |
| POST   | `/predict` | Upload gambar → hasil prediksi   |

### POST `/predict`

**Request:** `multipart/form-data`
- `file` — file gambar JPEG/PNG, maks. 10 MB

**Response:**
```json
{
  "filename": "xray.jpg",
  "mode": "dummy",
  "label": "PNEUMONIA",
  "confidence": 0.91,
  "probabilities": {
    "NORMAL": 0.09,
    "PNEUMONIA": 0.91
  }
}
```

**Mode:**
- `dummy` — model belum tersedia, hasil acak untuk testing
- `real` — model sudah di-load dari `models/pneumonia_cnn.pth`

## Menambahkan Model

Setelah training selesai, letakkan file model di:
```
api/models/pneumonia_cnn.pth
```

Model akan otomatis di-load saat server start. Format model: **ResNet-18** dengan output 2 kelas (`NORMAL`, `PNEUMONIA`).

## Testing

Swagger UI (browser): `http://127.0.0.1:8000/docs`

Via curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/ke/xray.jpg"
```
