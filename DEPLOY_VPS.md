# Deploy Scientia ke VPS

Dokumen ini menjelaskan cara deploy aplikasi Scientia di VPS Linux. Arsitektur deploy:

- Frontend static HTML/CSS/JS diserve oleh Nginx.
- Backend FastAPI berjalan dengan Uvicorn sebagai systemd service.
- Nginx reverse proxy request `/api/*` ke FastAPI lokal.
- Model checkpoint dibaca dari `api/models/checkpoints/inference_model.pth`.

Contoh domain di dokumen ini memakai `scientia.example.com`. Ganti dengan domain atau IP VPS milikmu.

## 1. Kebutuhan Server

Rekomendasi minimum:

- Ubuntu 22.04/24.04 LTS
- RAM minimal 2 GB, disarankan 4 GB karena PyTorch cukup berat
- Storage minimal 10 GB kosong
- Python 3.11
- Nginx
- Git

Login ke VPS:

```bash
ssh root@YOUR_SERVER_IP
```

Update package:

```bash
apt update
apt upgrade -y
```

Install dependency sistem:

```bash
apt install -y git nginx python3.11 python3.11-venv python3-pip ufw
```

Jika `python3.11` tidak tersedia di repository VPS, gunakan Python 3.10 atau 3.12 yang didukung oleh versi PyTorch di `api/requirements.txt`. Hindari Python 3.14 untuk project ini.

## 2. Ambil Source Code

Buat folder aplikasi:

```bash
mkdir -p /var/www/scientia
cd /var/www/scientia
```

Clone repo:

```bash
git clone YOUR_REPOSITORY_URL .
```

Pastikan file model ada:

```bash
ls -lh api/models/checkpoints/
```

Minimal harus ada salah satu:

```text
inference_model.pth
best_model.pth
```

File prioritas untuk API adalah:

```text
api/models/checkpoints/inference_model.pth
```

## 3. Setup Backend FastAPI

Buat virtual environment:

```bash
cd /var/www/scientia
python3.11 -m venv api/.venv
```

Install dependency:

```bash
api/.venv/bin/pip install --upgrade pip
api/.venv/bin/pip install -r api/requirements.txt
```

Test import dependency:

```bash
api/.venv/bin/python -c "import torch, torchvision, fastapi; print(torch.__version__, torchvision.__version__, fastapi.__version__)"
```

Test load API dan model:

```bash
cd /var/www/scientia/api
.venv/bin/python -c "import app; print(app.health())"
```

Target output:

```json
{"status":"ok","model_loaded":true,"mode":"real"}
```

Jika `model_loaded` masih `false`, cek bagian troubleshooting di bawah.

## 4. Environment Backend

Buat file env:

```bash
nano /var/www/scientia/api/.env
```

Isi:

```env
CHECKPOINT_DIR=models/checkpoints
DEVICE=cpu
THRESHOLD=0.5
NUM_WORKERS=2
```

Untuk VPS tanpa GPU, tetap gunakan:

```env
DEVICE=cpu
```

## 5. Buat systemd Service

Buat service:

```bash
nano /etc/systemd/system/scientia-api.service
```

Isi:

```ini
[Unit]
Description=Scientia Pneumonia Detection FastAPI
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/scientia/api
EnvironmentFile=/var/www/scientia/api/.env
ExecStart=/var/www/scientia/api/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Berikan permission folder ke `www-data`:

```bash
chown -R www-data:www-data /var/www/scientia
```

Start service:

```bash
systemctl daemon-reload
systemctl enable scientia-api
systemctl start scientia-api
```

Cek status:

```bash
systemctl status scientia-api --no-pager
```

Cek log:

```bash
journalctl -u scientia-api -f
```

Test backend lokal di VPS:

```bash
curl http://127.0.0.1:8000/health
```

Target:

```json
{"status":"ok","model_loaded":true,"mode":"real"}
```

## 6. Sesuaikan Frontend untuk Production

Frontend saat development memakai:

```js
const API_BASE = "http://127.0.0.1:8000";
```

Di VPS, browser user tidak bisa akses `127.0.0.1:8000` milik server. Ubah agar memakai path reverse proxy Nginx:

```bash
nano /var/www/scientia/frontend/js/app.js
```

Ubah baris pertama menjadi:

```js
const API_BASE = "/api";
```

Dengan konfigurasi ini, request frontend ke:

```text
/api/predict
```

akan diteruskan Nginx ke:

```text
http://127.0.0.1:8000/predict
```

## 7. Konfigurasi Nginx

Buat config:

```bash
nano /etc/nginx/sites-available/scientia
```

Isi untuk domain:

```nginx
server {
    listen 80;
    server_name scientia.example.com;

    root /var/www/scientia/frontend;
    index index.html;

    client_max_body_size 10M;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 30s;
        proxy_send_timeout 120s;
    }
}
```

Jika belum punya domain dan memakai IP langsung:

```nginx
server_name _;
```

Aktifkan site:

```bash
ln -s /etc/nginx/sites-available/scientia /etc/nginx/sites-enabled/scientia
```

Opsional: hapus default site:

```bash
rm -f /etc/nginx/sites-enabled/default
```

Test config:

```bash
nginx -t
```

Reload Nginx:

```bash
systemctl reload nginx
```

## 8. Firewall

Aktifkan firewall:

```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
ufw status
```

Jangan expose port `8000` ke publik. FastAPI cukup berjalan di `127.0.0.1:8000`, lalu publik masuk lewat Nginx port `80/443`.

## 9. HTTPS dengan Let's Encrypt

Jika memakai domain, install Certbot:

```bash
apt install -y certbot python3-certbot-nginx
```

Generate SSL:

```bash
certbot --nginx -d scientia.example.com
```

Test auto-renew:

```bash
certbot renew --dry-run
```

Setelah HTTPS aktif, akses:

```text
https://scientia.example.com
```

## 10. Test End-to-End

Test root frontend:

```bash
curl -I http://scientia.example.com
```

Test backend via Nginx:

```bash
curl http://scientia.example.com/api/health
```

Target:

```json
{"status":"ok","model_loaded":true,"mode":"real"}
```

Test upload gambar:

```bash
curl -X POST http://scientia.example.com/api/predict \
  -F "file=@/path/to/xray.jpg"
```

Contoh response:

```json
{
  "filename": "xray.jpg",
  "mode": "real",
  "label": "PNEUMONIA",
  "confidence": 0.9234,
  "probabilities": {
    "NORMAL": 0.0766,
    "PNEUMONIA": 0.9234
  },
  "inference_time_ms": 45.23
}
```

## 11. Update Deployment

Saat ada update dari Git:

```bash
cd /var/www/scientia
git pull origin feat/web-app
```

Jika dependency berubah:

```bash
api/.venv/bin/pip install -r api/requirements.txt
```

Jika frontend JS/CSS berubah, reload Nginx biasanya tidak wajib karena file static langsung dibaca. Untuk backend:

```bash
systemctl restart scientia-api
```

Cek ulang:

```bash
curl http://127.0.0.1:8000/health
curl http://scientia.example.com/api/health
```

## 12. Troubleshooting

### API mode masih dummy

Cek log:

```bash
journalctl -u scientia-api -n 100 --no-pager
```

Penyebab umum:

- `torch` belum terinstall di venv.
- File checkpoint tidak ada.
- `CHECKPOINT_DIR` salah.
- Permission folder model tidak bisa dibaca oleh `www-data`.

Cek:

```bash
ls -lh /var/www/scientia/api/models/checkpoints/
/var/www/scientia/api/.venv/bin/python -c "import torch; print(torch.__version__)"
```

### API mencoba download DenseNet dari internet

Pastikan file [api/models/model.py](api/models/model.py) saat load checkpoint memakai:

```python
pretrained=False
```

Saat inference, arsitektur tidak perlu download ImageNet weights karena bobot model sudah ada di checkpoint lokal.

### Browser gagal connect ke API

Pastikan `frontend/js/app.js` production memakai:

```js
const API_BASE = "/api";
```

Jangan gunakan ini di VPS:

```js
const API_BASE = "http://127.0.0.1:8000";
```

Karena `127.0.0.1` di browser user berarti komputer user, bukan VPS.

### Upload gagal 413 Request Entity Too Large

Pastikan Nginx config punya:

```nginx
client_max_body_size 10M;
```

Lalu reload:

```bash
nginx -t
systemctl reload nginx
```

### Service gagal start karena permission

Perbaiki owner:

```bash
chown -R www-data:www-data /var/www/scientia
systemctl restart scientia-api
```

### Inference lambat

Di CPU, inference PyTorch bisa lambat pada VPS kecil. Solusi:

- Gunakan VPS RAM/CPU lebih besar.
- Pastikan memakai `inference_model.pth`, bukan checkpoint training penuh.
- Jalankan satu Uvicorn worker dulu. Model PyTorch besar akan dimuat per worker jika worker ditambah.

## 13. Checklist Final

Sebelum dianggap selesai:

- `curl http://127.0.0.1:8000/health` mengembalikan `mode: real`.
- `curl http://DOMAIN/api/health` mengembalikan `mode: real`.
- Frontend bisa dibuka dari domain/IP.
- Upload X-Ray dari browser berhasil.
- Gambar tetap tampil saat hasil analisis muncul.
- Port publik yang terbuka hanya `22`, `80`, dan/atau `443`.
- File `.env` dan `.venv` tidak ikut commit ke Git.
