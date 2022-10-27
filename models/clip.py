import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(
        self,
        temp,
        image_embedding_size,
        text_embedding_size,
        audio_encoder=ResNet50()
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = TextEncoder()
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

class InfoNCE(nn.Module):
    def __init__(self, temp) -> None:
        self.temperature = 1
        super().__init__()

    def forward(self, text_embeddings, audio_embeddings):
        logits = (text_embeddings @ audio_embeddings.T) / self.temperature
        audio_similarity = audio_embeddings @ audio_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (audio_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  images_loss + texts_loss # shape: (batch_size)
        return loss.mean()

    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()