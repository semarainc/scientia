import io
import os
import random
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(
    title="Scientia - Pneumonia Detection API",
    description="Binary classification API: PNEUMONIA vs NORMAL dari citra X-Ray dada",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["NORMAL", "PNEUMONIA"]

# ---------- model loader ----------
predictor = None

def load_model():
    """Load PneumoniaPredictor from models/predict.py"""
    global predictor
    try:
        from models.predict import get_predictor
        
        # Get checkpoint path from environment or use default
        checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "models/checkpoints"))
        checkpoint_path = checkpoint_dir / "inference_model.pth"
        
        # Fallback to best_model.pth if inference_model.pth doesn't exist
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / "best_model.pth"
        
        if not checkpoint_path.exists():
            print(f"[WARNING] Model checkpoint tidak ditemukan di: {checkpoint_path}")
            return False
        
        # Initialize predictor
        predictor = get_predictor(
            model_path=str(checkpoint_path),
            device=os.getenv("DEVICE", "cpu"),
            threshold=float(os.getenv("THRESHOLD", "0.5"))
        )
        print(f"[INFO] Model berhasil dimuat dari: {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"[WARNING] Gagal load model: {e}")
        import traceback
        traceback.print_exc()
        return False

load_model()

# ---------- inference ----------
def predict_dummy(image: Image.Image) -> dict:
    """Dummy inference — hapus dan ganti dengan predict_real() setelah model tersedia."""
    time.sleep(0.5)  # simulasi latency
    label_idx = random.randint(0, 1)
    confidence = round(random.uniform(0.70, 0.99), 4)
    return {
        "label": CLASSES[label_idx],
        "confidence": confidence,
        "probabilities": {
            "NORMAL": round(1 - confidence if label_idx == 1 else confidence, 4),
            "PNEUMONIA": round(confidence if label_idx == 1 else 1 - confidence, 4),
        },
    }

def predict_real(image: Image.Image) -> dict:
    """Real inference using PneumoniaPredictor"""
    if predictor is None:
        raise RuntimeError("Model predictor tidak tersedia")
    
    # Predict menggunakan predictor yang sudah di-load
    result = predictor.predict(image)
    
    # result dari predictor.predict() memiliki format:
    # {
    #     "label": "NORMAL" atau "PNEUMONIA",
    #     "label_index": 0 atau 1,
    #     "probability": float (P(PNEUMONIA)),
    #     "confidence": float,
    #     "threshold": float,
    #     "inference_ms": float
    # }
    
    # Konversi ke format yang diharapkan API
    prob_pneumonia = result["probability"]
    prob_normal = 1.0 - prob_pneumonia
    
    return {
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "probabilities": {
            "NORMAL": round(prob_normal, 4),
            "PNEUMONIA": round(prob_pneumonia, 4),
        },
        "inference_time_ms": result.get("inference_ms", 0),
    }

# ---------- endpoints ----------
@app.get("/")
def root():
    return {"message": "Scientia Pneumonia Detection API is running", "status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "mode": "real" if predictor is not None else "dummy",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG).")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File gambar tidak valid atau rusak.")

    if predictor is not None:
        result = predict_real(image)
    else:
        result = predict_dummy(image)

    return JSONResponse(content={
        "filename": file.filename,
        "mode": "real" if predictor is not None else "dummy",
        **result,
    })
