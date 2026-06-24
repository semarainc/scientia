import io
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

MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_cnn.pth"
CLASSES = ["NORMAL", "PNEUMONIA"]

# ---------- model loader ----------
model = None

def load_model():
    global model
    if not MODEL_PATH.exists():
        return False
    try:
        import torch
        import torchvision.models as tv_models

        net = tv_models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        net.eval()
        model = net
        return True
    except Exception as e:
        print(f"[WARNING] Gagal load model: {e}")
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
    import torch
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    label_idx = int(torch.argmax(torch.tensor(probs)).item())
    return {
        "label": CLASSES[label_idx],
        "confidence": round(max(probs), 4),
        "probabilities": {
            "NORMAL": round(probs[0], 4),
            "PNEUMONIA": round(probs[1], 4),
        },
    }

# ---------- endpoints ----------
@app.get("/")
def root():
    return {"message": "Scientia Pneumonia Detection API is running", "status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mode": "real" if model is not None else "dummy",
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

    if model is not None:
        result = predict_real(image)
    else:
        result = predict_dummy(image)

    return JSONResponse(content={
        "filename": file.filename,
        "mode": "real" if model is not None else "dummy",
        **result,
    })
