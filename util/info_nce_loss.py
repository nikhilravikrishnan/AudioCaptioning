import torch
import torch.nn as nn

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