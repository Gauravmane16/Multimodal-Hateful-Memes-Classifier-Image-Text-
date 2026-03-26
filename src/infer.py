import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from src.data import extract_ocr_text


def predict(model, image, caption, device="cpu", image_size=224):
    """
    Make a prediction on a single image and caption.
    
    Args:
        model: FusionModel instance
        image: PIL Image or path to image
        caption: str, caption text
        device: torch device
        image_size: int, size for image transforms
    
    Returns:
        label: str, "hateful" or "not_hateful"
        confidence: float, probability of being hateful
    """
    model.eval()
    
    # Load and process image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process text with OCR
    if isinstance(image, str):
        ocr_text = extract_ocr_text(image)
    else:
        ocr_text = ""
    
    text = f"{caption} [OCR] {ocr_text}"
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        prob = torch.sigmoid(logits.squeeze()).item()
    
    label = "hateful" if prob >= 0.5 else "not_hateful"
    
    return label, prob
