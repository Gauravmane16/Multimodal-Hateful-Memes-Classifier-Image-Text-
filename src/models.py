import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            # Newer torchvision versions use the `weights=` API
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback for older torchvision versions
            base = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 2048

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.out_dim = 384

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


class FusionModel(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=384, hidden_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(self.image_encoder.out_dim + self.text_encoder.out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.image_encoder.out_dim + self.text_encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, input_ids, attention_mask):
        img_emb = self.image_encoder(image)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        logits = self.fusion(torch.cat([img_emb, txt_emb], dim=1))
        return logits
