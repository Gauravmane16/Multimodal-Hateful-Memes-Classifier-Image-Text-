import os
import json
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm

from src.models import FusionModel
from src.data import MemeDataset, load_splits, get_transforms
from src.utils import set_seed
from src.eval import evaluate_model


def train_epoch(model, train_loader, optimizer, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['loss']['pos_weight']]).to(device))
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        
        # Forward pass with AMP
        with autocast():
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits.squeeze(), label)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main():
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path(config['paths']['model_ckpt']).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    train_df, val_df, test_df = load_splits()
    
    # Get transforms
    train_transform, val_transform = get_transforms(config['data']['image_size'])
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Create datasets
    train_dataset = MemeDataset(train_df, tokenizer, train_transform)
    val_dataset = MemeDataset(val_df, tokenizer, val_transform)
    test_dataset = MemeDataset(test_df, tokenizer, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = FusionModel().to(device)
    
    # Optimizer and scaler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])
    )
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = config['training']['early_stopping_patience']
    
    print(f"\nTraining for {config['training']['epochs']} epochs...\n")
    
    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Val Accuracy: {val_metrics['accuracy']*100:.2f}%")
        print(f"Val ROC-AUC: {val_metrics['roc_auc']*100:.2f}%")
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config['paths']['model_ckpt'])
            print(f"✓ Model saved to {config['paths']['model_ckpt']}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    # Load best model
    print("\n" + "="*60)
    print("Loading best model for final evaluation...")
    if os.path.exists(config['paths']['model_ckpt']):
        model.load_state_dict(torch.load(config['paths']['model_ckpt']))
    else:
        print(f"Warning: Checkpoint {config['paths']['model_ckpt']} not found. Skipping model loading.")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {test_metrics['f1']*100:.2f}%")
    print(f"ROC-AUC:   {test_metrics['roc_auc']*100:.2f}%")
    print("\nConfusion Matrix (%):")
    print(test_metrics['confusion_matrix'])
    print("="*60)
    
    # Save metrics
    metrics_path = "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.number))
                   else v.tolist() if isinstance(v, np.ndarray)
                   else v
                   for k, v in test_metrics.items()}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    return test_metrics


if __name__ == "__main__":
    main()