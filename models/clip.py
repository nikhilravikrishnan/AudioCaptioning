import torch
import torch.nn as nn
import text_encoder
import audio_encoders
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, ViTFeatureExtractor

class ViTClip(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=ViTModel.from_pretrained("google/vit-base-patch16-224-in21k"),
        feature_extractor=ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder.TextEncoder()
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size)
        self.temperature = temp
        self.audio_embeddings = None
        self.text_embeddings = None
        self.feature_extractor = feature_extractor
        self.set_hook = True

    def forward(self, batch):
        #Getting audio and text features
        # In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W)
        # See: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTFeatureExtractor.__call__
        audio_features = None
        image_features = self.feature_extractor(batch["image"], return_tensors="pt")
        with torch.no_grad():
            raw_audio_features = self.audio_encoder(image_features)
            audio_features = raw_audio_features.last_hidden_state[:, 0, :]
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)
        return audio_embeddings, text_embeddings

class BaseClip(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        layer_dict={"avgpool":"features"}
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder.TextEncoder()
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size)
        self.temperature = temp
        self.audio_embeddings = None
        self.text_embeddings = None
        self.layer_dict = layer_dict

    def forward(self, batch):
        # Getting audio and text features
        raw_audio_features = batch["image"]
        with torch.no_grad():
            audio_features = audio_encoders.get_audio_feature_vector(self.audio_encoder, raw_audio_features, self.layer_dict)
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)
        return audio_embeddings, text_embeddings

class PANNClip(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=audio_encoders.Cnn14(), #add default params
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
        raw_audio_features = batch["image"]
        with torch.no_grad():
            audio_features = self.audio_encoder(raw_audio_features)
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features["embeddings"])
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