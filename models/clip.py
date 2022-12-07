import sys
sys.path.append("/content/drive/My Drive/MusicCaptioning/Walter/util/")
sys.path.append("/content/drive/My Drive/MusicCaptioning/Walter/models/")
from util.loss import InfoNCE

import torch
import torch.nn as nn
import text_encoder
import audio_encoders
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
#import ntlk
import time

import numpy as np

class ViTClip(nn.Module):
    def __init__(
        self,
        device,
        image_embedding_size=768,
        text_embedding_size=768, 
        audio_encoder=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
        fine_tune=False
        ):
        super().__init__()
        self.audio_encoder = audio_encoder.to(device)
        self.text_encoder = text_encoder.TextEncoder().to(device)
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)
        self.transforms = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        self.fine_tune = fine_tune
        self.device = device
        self.audio_size = image_embedding_size
        self.layer_extract = ["heads.head", "encoder.layers.encoder_layer_11.mlp.4", "encoder.ln"] # Layer to use as the audio embedding
        self._features = {layer: torch.empty(0) for layer in self.layer_extract}
        
        
        # Registering a forward hook to save output for the audio embeddings
        for layer_id in self.layer_extract:
            layer = dict([*self.audio_encoder.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_output_hook(layer_id))

        if fine_tune == True:
            # The list of layers to fine tune
            params = ["audio_encoder.heads", "audio_encoder.encoder.layers.encoder_layer_11", 
            "audio_encoder.encoder.layers.encoder_layer_10", "audio_projection", "text_projection"]
            
            # Only create a gradient in the computational graph for the selected layers
            for name, param in self.named_parameters():
                for p in params:
                    if p in name:
                        param.requires_grad = True
                        break
                    else:
                        param.requires_grad = False

    def save_output_hook(self, layer):
        # Forward hook function is of the form:
        # hook(module, input, output) -> None or modified output
        # Here we use it to set the self._features[layer] property on the forward pass
        def fn(_, __, output):
            self._features[layer] = output
        return fn
    
    def forward(self, batch):
        from util.loss import InfoNCE
        loss = InfoNCE()

        #Getting audio and text features
        raw_audio_features = batch[0]
        raw_ids = batch[1]
        raw_mask = batch[2]

        batch_size = raw_audio_features.shape[0]
        
        # Reshaping each data tensor into the appropriate shape
        raw_ids = raw_ids.reshape(-1, raw_ids.shape[-1])
        raw_mask = raw_mask.reshape(-1, raw_mask.shape[-1])

        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=raw_ids, attention_mask=raw_mask
            )

        processed_audio = torch.zeros((batch_size, self.audio_size))

        if self.fine_tune == False:
            with torch.no_grad():
                for i in range(raw_audio_features.size()[0]):
                    transformed_audio = self.transforms(raw_audio_features[i, :, :, :].squeeze(0))
                    audio_features = audio_encoders.get_vit_feature_vector(self.audio_encoder, self.device, transformed_audio.type(torch.DoubleTensor))
                    processed_audio[i, :] = audio_features[0,:]
        else:
            for i in range(raw_audio_features.size()[0]):
                transformed_audio = self.transforms(raw_audio_features[i, :, :, :].unsqueeze(0))
                _ = self.audio_encoder(transformed_audio)
                audio_features = self._features["encoder.layers.encoder_layer_11.mlp.4"][0, :]
                processed_audio[i, :] = audio_features[0,:]
        
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(processed_audio.type(torch.FloatTensor).to(self.device))
        text_embeddings = self.text_projection(text_features)
        
        batch_loss = loss.forward(text_embeddings, audio_embeddings)
        
        return batch_loss, audio_embeddings, text_embeddings


class BaseClip(nn.Module):
    def __init__(
        self,
        device,
        image_embedding_size=1000,
        text_embedding_size=768,
        audio_encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        fine_tune=False
    ):
        super().__init__()
        self.audio_encoder = audio_encoder.to(device)
        self.text_encoder = text_encoder.TextEncoder().to(device)
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)
        self.device = device
        self.layer_extract = ["fc"] # Layer to use as the audio embedding
        self._features = {layer: torch.empty(0) for layer in self.layer_extract}
        self.fine_tune = fine_tune
        self.audio_size = image_embedding_size

        # Registering a forward hook to save output for the audio embeddings
        for layer_id in self.layer_extract:
            layer = dict([*self.audio_encoder.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_output_hook(layer_id))

        if fine_tune == True:
            # The list of layers to fine tune
            params = ["audio_encoder.layer4.0", 
                      "audio_encoder.layer4.1", 
                      "audio_encoder.layer4.2", 
                      "audio_encoder.fc"]
            
            # Only create a gradient in the computational graph for the selected layers
            for name, param in self.named_parameters():
                for p in params:
                    if p in name:
                        param.requires_grad = True
                        break
                    else:
                        param.requires_grad = False

    def save_output_hook(self, layer):
        # Forward hook function is of the form:
        # hook(module, input, output) -> None or modified output
        # Here we use it to set the self._features[layer] property on the forward pass
        def fn(_, __, output):
            self._features[layer] = output
        return fn

    def forward(self, batch):
        from util.loss import InfoNCE
        loss = InfoNCE()

        # Getting audio and text features
        raw_audio_features = batch[0]
        raw_ids = batch[1]
        raw_mask = batch[2]

        # Reshape raw_ids and raw_mask to (batch_size, seq_len)
        raw_ids = raw_ids.reshape(-1, raw_ids.shape[-1])
        raw_mask = raw_mask.reshape(-1, raw_mask.shape[-1])
        
        # Batch size
        batch_size = raw_audio_features.shape[0]

        text_features = self.text_encoder(
            input_ids=raw_ids, attention_mask=raw_mask
        )

        processed_audio = torch.zeros((batch_size, self.audio_size))

        if self.fine_tune == False:
            with torch.no_grad():
                for i in range(raw_audio_features.size()[0]):
                    _ = self.audio_encoder(raw_audio_features[i, :, :, :].unsqueeze(0))
                    audio_features = self._features["fc"]
                    processed_audio[i, :] = audio_features
        else:
            for i in range(raw_audio_features.size()[0]):
                _ = self.audio_encoder(raw_audio_features[i, :, :, :].unsqueeze(0))
                audio_features = self._features["fc"]
                processed_audio[i, :] = audio_features

        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(processed_audio.to(self.device))
        text_embeddings = self.text_projection(text_features)

        batch_loss = loss.forward(text_embeddings, audio_embeddings)

        return batch_loss, audio_embeddings, text_embeddings


class PANNClip(nn.Module):
    def __init__(
        self,
        device,
        temp=1,
        image_embedding_size=2048,
        text_embedding_size=768,
        audio_encoder=audio_encoders.Cnn14(), #add default params
        model_path="/content/drive/MyDrive/MusicCaptioning/Nikhil/models/pretrained_weights/Cnn14_mAP=0.431.pth",
        fine_tune=False

    ):
        super().__init__()
        self.audio_encoder = audio_encoder.to(device)
        self.text_encoder = text_encoder.TextEncoder().to(device)
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding_size).to(device)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_size).to(device)
        self.temperature = temp
        self.saved_model = torch.load(model_path)

        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_real.weight')
        self.saved_model['model'].pop('spectrogram_extractor.stft.conv_imag.weight')
        self.saved_model['model'].pop('logmel_extractor.melW')
        self.saved_model['model'].pop("fc1.weight")
        self.saved_model['model'].pop("fc1.bias")
        self.saved_model['model'].pop("fc_audioset.weight") 
        self.saved_model['model'].pop("fc_audioset.bias")
        self.device = device
        self.fine_tune = fine_tune

        self.audio_encoder.load_state_dict(self.saved_model["model"])

        self.audio_embeddings = None
        self.text_embeddings = None

    def forward(self, batch):

        loss = InfoNCE()

        raw_audio_features = batch[0]

        audio_features = self.audio_encoder(raw_audio_features)

        raw_ids = batch[1]
        raw_mask = batch[2]

        # Reshape raw_ids and raw_mask to (batch_size, seq_len)
        raw_ids = raw_ids.reshape(-1, raw_ids.shape[-1])
        raw_mask = raw_mask.reshape(-1, raw_mask.shape[-1])

        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=raw_ids, attention_mask=raw_mask
            )
        # Getting audio and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)

        batch_loss = loss.forward(text_embeddings, audio_embeddings)

        return batch_loss, audio_embeddings, text_embeddings


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

if __name__ == "__main__":
    print(sys.path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_model = ViTClip(device, fine_tune=True)
    
    

