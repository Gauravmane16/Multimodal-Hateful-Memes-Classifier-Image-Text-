import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        # Move to device
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        
        # Forward pass
        logits = model(image, input_ids, attention_mask)
        
        # Get predictions
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_preds.append(np.array(preds).reshape(-1))
        all_probs.append(np.array(probs).reshape(-1))
        all_labels.append(label.cpu().numpy().reshape(-1))
    
    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics (in decimal form, not %)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = float('nan')
    # Confusion matrix (normalized by true class, in %)
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


def evaluate(y_true, y_prob):
    """Legacy evaluation function."""
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob) * 100

    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100

    return {
        "accuracy": acc,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }


def get_misclassified_examples(model, dataloader, device, num_examples=10):
    """Get misclassified examples from dataloader."""
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            
            logits = model(image, input_ids, attention_mask)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Find misclassified
            for i in range(len(preds)):
                if preds[i] != int(label[i].item()):
                    misclassified.append({
                        "image": image[i].cpu(),
                        "input_ids": input_ids[i].cpu(),
                        "true_label": int(label[i].item()),
                        "pred_label": int(preds[i]),
                        "confidence": float(probs[i])
                    })
                    
                    if len(misclassified) >= num_examples:
                        return misclassified
    
    return misclassified
