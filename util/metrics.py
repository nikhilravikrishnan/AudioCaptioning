import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from dataloaders.clotho_dataloader import AudioCaptioningDataset

def mean_reciprocal_rank(model, audio_embeddings, caption_embeddings):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
    output: - mean_reciprocal_rank: Scalar.
    """

    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = caption_embeddings @ audio_embeddings.T

    return ((1/torch.argmax(cosine_similarity, dim=1)).sum() / cosine_similarity.shape[0]).item()

# Implement mean reciprocal rank metric for evaluation

def mAPatK():
    return

def recallAtK():
    return

def evaluate_model(model: nn.Module, checkpoint_path: str, metrics: list, data_dir: str, split="eval"):
    """
    Load and evaluate a model using the specified metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.load_state_dict(torch.load(checkpoint_path), map_location=device)
    
    dataset = AudioCaptioningDataset(data_dir = data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    results = {"mrr":[]}

    for (idx,batch) in enumerate(dataloader):
        text_emb = None
        audio_emb = None
        # Send batch to device
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        # Get the embeddings from the model
        _, audio_emb, text_emb = model(batch)

        if "recall" in metrics:
            pass
        if "precision" in metrics:
            pass
        if "mrr" in metrics:
            mrr = mean_reciprocal_rank(model, audio_emb, text_emb)
            results["mrr"].append(mrr)
            print(mrr)

    return results