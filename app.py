from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import torch
import yaml
from src.models import FusionModel
from src.infer import predict

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel().to(device)
model.load_state_dict(torch.load(config['paths']['model_ckpt'], map_location=device))
model.eval()

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hateful Memes Detection API",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict_api(image: UploadFile = File(...), caption: str = Form("")):
    
    # Validate file
    if image.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Invalid file type"}

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    label, confidence = predict(model, img, caption, device)

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
