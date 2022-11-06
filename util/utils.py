import torch
import torch.nn.functional as F

def load_pretrained_img_model(model, device, checkpoint_path):
    pretrained_model = model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model"])
    return pretrained_model


def eval_model_embeddings(model, dataLoader, metric_name: str, text_query: str,  **kwargs):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model: CLIP model
            - metric_name: ['MRR', 'MAP@K', 'R@K']
    output: - metric specified
    """
    ret = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio_embedding_list = []
    text_embedding_list = []

    for (idx, batch) in enumerate(dataLoader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        _, audio_encoders, text_encoders = model.forward(batch)
        audio_embedding_list.append([audio_encoders])
        text_embedding_list.append([text_encoders])
    
    # Concatenate all the embeddings
    audio_embeddings = torch.cat(audio_embedding_list, dim = 0)
    text_embeddings = torch.cat(text_embedding_list, dim = 0)

    if metric_name == 'MRR':
        ret = mean_reciprocal_rank(audio_embeddings, text_embeddings)
    
    if metric_name == 'MAP@K':
        if 'k' in kwargs:
            raise ValueError("Needs K parameter.")

        ret = mean_avg_precision_at_k(audio_embeddings, text_embeddings, k = kwargs['k'])

    if metric_name == 'R@K':
        if 'k' in kwargs:
            raise ValueError("Needs K parameter.")

        ret = mean_reciprocal_rank(audio_embeddings, text_embeddings, k = kwargs['k'])

    if metric_name == 'None':
        raise NotImplementedError

    return ret

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

def mean_avg_precision_at_k(model, audio_embeddings, caption_embeddings, k=10):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
            - k: Scalar
    output: - mean_avg_precision_at_k: Scalar.
    
    """
    ret = None
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = caption_embeddings @ audio_embeddings.T

    # Find unique audio embeddings
    unique_audio_embeddings = torch.unique(audio_embeddings, dim=0)
    # Find indices for each unique audio embedding
    unique_audio_embedding_indices = [torch.where(torch.all(audio_embeddings == unique_audio_embeddings[i], dim=1))[0] for i in range(unique_audio_embeddings.shape[0])]

    # Create zero like tensor same shape as cosine similarity
    cosine_similarity_mask = torch.zeros_like(cosine_similarity)
    
    # For each unique audio embedding, compute the combinations of indices taken 2 at a time
    for i in range(len(unique_audio_embedding_indices)):
        for j in range(len(unique_audio_embedding_indices[i])):
            for k in range(j, len(unique_audio_embedding_indices[i])):
                cosine_similarity_mask[unique_audio_embedding_indices[i][j], unique_audio_embedding_indices[i][k]] = 1
                cosine_similarity_mask[unique_audio_embedding_indices[i][k], unique_audio_embedding_indices[i][j]] = 1

    # Sort cosine similarity in descending order row wise and get the indices
    cosine_similarity_sorted, cosine_similarity_sorted_indices = torch.sort(cosine_similarity, dim=1, descending=True)
    # Sort cosine similarity mask row wise using the indices from above
    cosine_similarity_mask_sorted = cosine_similarity_mask[torch.arange(cosine_similarity_mask.shape[0]).unsqueeze(1), cosine_similarity_sorted_indices]
    # Sum over the first k columns of the sorted cosine similarity mask and divide by k
    ret = (cosine_similarity_mask_sorted[:, :k].sum(dim=1) / k).mean().item()

    return ret


def mean_recall_at_k(model, audio_embeddings, caption_embeddings, k=10):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
            - k: Scalar
    output: - recall_at_k: Scalar.
    
    """
    ret = None
    
    ret = None
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = caption_embeddings @ audio_embeddings.T

    # Find unique audio embeddings
    unique_audio_embeddings = torch.unique(audio_embeddings, dim=0)
    # Find indices for each unique audio embedding
    unique_audio_embedding_indices = [torch.where(torch.all(audio_embeddings == unique_audio_embeddings[i], dim=1))[0] for i in range(unique_audio_embeddings.shape[0])]

    # Create zero like tensor same shape as cosine similarity
    cosine_similarity_mask = torch.zeros_like(cosine_similarity)
    
    # For each unique audio embedding, compute the combinations of indices taken 2 at a time
    for i in range(len(unique_audio_embedding_indices)):
        for j in range(len(unique_audio_embedding_indices[i])):
            for k in range(j, len(unique_audio_embedding_indices[i])):
                cosine_similarity_mask[unique_audio_embedding_indices[i][j], unique_audio_embedding_indices[i][k]] = 1
                cosine_similarity_mask[unique_audio_embedding_indices[i][k], unique_audio_embedding_indices[i][j]] = 1

    # Sort cosine similarity in descending order row wise and get the indices
    cosine_similarity_sorted, cosine_similarity_sorted_indices = torch.sort(cosine_similarity, dim=1, descending=True)
    # Sort cosine similarity mask row wise using the indices from above
    cosine_similarity_mask_sorted = cosine_similarity_mask[torch.arange(cosine_similarity_mask.shape[0]).unsqueeze(1), cosine_similarity_sorted_indices]
    # Sum over the first k columns of the sorted cosine similarity mask and divide by k
    ret = (cosine_similarity_mask_sorted[:, :k].sum(dim=1) / cosine_similarity_mask.sum(dim=1)).mean().item()

    return ret