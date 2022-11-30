import torch
import torch.nn as nn
import text_encoder
import audio_encoders
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, ViTFeatureExtractor
from util.loss import InfoNCE

import numpy as np

class ViTClip(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size=768,
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
        image_embedding_size=2048,
        text_embedding_size=768,
        audio_encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        layer_dict={"avgpool":"features"},
        vocab_file="/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data/words_list.p"
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
        self.vocab_file = vocab_file

    def forward(self, batch):

        loss = InfoNCE()


        # Getting audio and text features
        raw_audio_features = batch[0]
        raw_ids = batch[1]
        raw_mask = batch[2]

        # Reshape raw_ids and raw_mask to (batch_size, seq_len)
        raw_ids = raw_ids.reshape(-1, raw_ids.shape[-1])
        raw_mask = raw_mask.reshape(-1, raw_mask.shape[-1])
        
        processed_audio = []
        with torch.no_grad():
            for i in range(raw_audio_features.size()[0]):
                audio_features = audio_encoders.get_audio_feature_vector(self.audio_encoder, raw_audio_features[i, :, :, :], self.layer_dict)
                processed_audio.append(audio_features)
        audio_stack = torch.stack(processed_audio)
        
        text_features = self.text_encoder(
            input_ids=raw_ids, attention_mask=raw_mask
        )

        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_stack)
        text_embeddings = self.text_projection(text_features)

        batch_loss = loss.forward(text_embeddings, audio_embeddings)

        return batch_loss, audio_embeddings, text_embeddings

class PANNClip(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size=2048,
        text_embedding_size=768,
        audio_encoder=audio_encoders.Cnn14(), #add default params
        model_path="/home/nikhilrk/MusicCaptioning/MusicCaptioning/models/pretrained_weights/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder.TextEncoder()
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size)
        self.temperature = temp
        self.saved_model = torch.load(model_path)

        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_real.weight')
        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_imag.weight')
        self.saved_model['model'].pop('logmel_extractor.melW')
        self.saved_model['model'].pop("fc1.weight")
        self.saved_model['model'].pop("fc1.bias")
        self.saved_model['model'].pop("fc_audioset.weight") 
        self.saved_model['model'].pop("fc_audioset.bias")


        audio_encoder.load_state_dict(self.saved_model["model"])


        self.audio_embeddings = None
        self.text_embeddings = None

    def forward(self, batch):
        raw_audio_features = batch["image"]
        processed_audio = []
        
        
        audio_stack = torch.stack(processed_audio)
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_stack)
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
        self.projection = nn.Linear(embedding_dim, projection_dim).double()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(projection_dim, projection_dim).double()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim).double()
    
    def forward(self, x):
        
        # Reshape x to (batch_size, seq_len, embedding_dim)
        x = x.reshape(-1, x.shape[-1]).double()

        projected = self.projection(x)
        x = self.relu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # Residual
        x = self.layer_norm(x)
        
        return x


