import torch
import torch.nn as nn
import text_encoder
import audio_encoders
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel

class CLIPModel(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        feature_extractor=None
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

    def forward(self, batch, layer_dict=None):
        # Getting audio and text features
        image_features = batch["image"]
        # Used for Image Transformer model
        if self.feature_extractor is not None:
            # In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W)
            # See: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTFeatureExtractor.__call__
            image_features = self.feature_extractor(batch["image"], return_tensors="pt")
        raw_audio_features = None
        processed_audio_features = None
        # This is used for ResNet and other similar torchvision models
        if self.set_hook == True:
            processed_audio_features = audio_encoders.get_audio_feature_vector(self.audio_encoder, image_features, layer_dict)
        # This is used for everything else
        else:
            with torch.no_grad():
                raw_audio_features = self.audio_encoder(image_features)
        # Process the audio features if needed (Vision Transformer, PANN)
        processed_audio_features = self.process_raw_audio_feat(raw_audio_features)
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(processed_audio_features)
        text_embeddings = self.text_projection(text_features)
        return audio_embeddings, text_embeddings

    def process_raw_audio_feat(self, raw_audio_features):
        if self.audio_encoder == ViTModel():
            # Get the output for the first (CLS) token for each sample
            return raw_audio_features.last_hidden_state[:, 0, :]
        elif self.audio_encoder == audio_encoders.Wavegram_Logmel_Cnn14():
            # Get the "embedding" tensors from the output dict
            return raw_audio_features["embedding"]
        else:
            raise NotImplementedError
        return

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