import torch
import torch.nn as nn
import text_encoder
from torchvision.models import resnet50, ResNet50_Weights

class CLIPModel(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder.TextEncoder()
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size)
        self.temperature = temp
        self.audio_embeddings = None
        self.text_embeddings = None

    def forward(self, batch):
        # Getting audio and text features
        audio_features = self.audio_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)
        return audio_embeddings, text_embeddings

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # Residual
        x = self.layer_norm(x)
        return x