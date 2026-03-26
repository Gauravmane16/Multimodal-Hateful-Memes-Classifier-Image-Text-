import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

import easyocr
import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

reader = easyocr.Reader(['en'])


def extract_ocr_text(image_path):
    """Extract text from image using EasyOCR."""
    try:
        result = reader.readtext(image_path, detail=0)
        return " ".join(result)
    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""


def map_sentiment_to_label(sentiment):
    sentiment = sentiment.strip().lower()
    if sentiment == "not_hateful":
        return 0
    elif "hateful" in sentiment:
        return 1
    else:
        return 0  # Default to NOT_HATEFUL if unknown

class MemeDataset(Dataset):
    """Dataset for multimodal hateful meme classification."""
    
    def __init__(self, df, tokenizer, img_transform):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and transform image
        image_path = f"data/raw/{row['image_name']}"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Use corrected text
        text = row['text_corrected']

        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "label": torch.tensor(row["label"], dtype=torch.float)
        }


def load_splits(splits_path="data/splits.json", seed=42):
    """Load dataset from flat JSON and split into train/val/test, fallback to random split if needed. Print class counts."""
    with open(splits_path, 'r') as f:
        data = json.load(f)

    # Map sentiment to label
    for item in data:
        item["label"] = map_sentiment_to_label(item["overall_sentiment"]) 

    df = pd.DataFrame(data)

    # Print class distribution
    print("Class distribution:")
    print(df['label'].value_counts())

    try:
        # Try stratified split
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])
    except ValueError as e:
        print(f"Stratified split failed: {e}\nFalling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, shuffle=True, stratify=None)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True, stratify=None)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_transforms(image_size=224):
    """Get image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform
