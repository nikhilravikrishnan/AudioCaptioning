import torch
import torch.nn.functional as F

def load_pretrained_img_model(model, device, checkpoint_path):
    pretrained_model = model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model"])
    
    return pretrained_model

def eval_model_embeddings(model, device, dataLoader, metric_name: list, **kwargs):
    """
    Obtain the evaluation metric of the specified type from the given model
    input:  - model: CLIP model
            - metric_name: a list containing 1 or more of the following: ['MRR', 'MAP@K', 'R@K']
    output: - metric specified
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    metrics = {"MRR":[], "MAP@K":[], "R@K":[]}

    for (idx, batch) in enumerate(dataLoader):
        print(f"Calculating metrics for batch {idx}...")
        
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        _, audio_embeddings, text_embeddings = model.forward(batch)

        if 'MRR' in metric_name:
            #print("Calculating MRR...")
            metrics["MRR"].append(mean_reciprocal_rank(audio_embeddings, text_embeddings))

        if 'MAP@K' in metric_name:
            if 'k' not in kwargs:
                raise ValueError("Needs K parameter.")
            metrics["MAP@K"].append(mean_avg_precision_at_k(audio_embeddings, text_embeddings, k = kwargs['k']))

        if 'R@K' in metric_name:
            if 'k' not in kwargs:
                raise ValueError("Needs K parameter.")
            #print("Calculating R@K...")
            metrics["R@K"].append(mean_recall_at_k(audio_embeddings, text_embeddings, k = kwargs['k']))

    # Generate the mean for each metric across all batches
    for k in metrics.keys():
        metrics[k] = sum(metrics[k])/len(metrics[k])
            
    return metrics
   

def mean_reciprocal_rank(audio_embeddings, caption_embeddings):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
    output: - mean_reciprocal_rank: Scalar.
    """
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = torch.nn.functional.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = audio_embeddings @ caption_embeddings.T 

    # Find unique audio embeddings
    unique_audio_embeddings = torch.unique(audio_embeddings, dim=0)

    # Find indices for each unique audio embedding
    unique_audio_embedding_indices = [torch.where(torch.all(audio_embeddings == unique_audio_embeddings[i], dim=1))[0] for i in range(unique_audio_embeddings.shape[0])]

    # Create zero like tensor same shape as cosine similarity
    cosine_similarity_mask = torch.zeros_like(cosine_similarity)

    # For each unique audio embedding, compute the combinations of indices taken 2 at a time
    for i in range(len(unique_audio_embedding_indices)):
        for j in range(len(unique_audio_embedding_indices[i])):
            for h in range(j, len(unique_audio_embedding_indices[i])):
                cosine_similarity_mask[unique_audio_embedding_indices[i][j], unique_audio_embedding_indices[i][h]] = 1
                cosine_similarity_mask[unique_audio_embedding_indices[i][h], unique_audio_embedding_indices[i][j]] = 1

    # Sort cosine similarity in descending order row wise and get the indices
    cosine_similarity_sorted, cosine_similarity_sorted_indices = torch.sort(cosine_similarity, dim=1, descending=True)
    cosine_similarity_mask_sorted_indices = cosine_similarity_sorted_indices * cosine_similarity_mask

    # Get the rank of the first non-zero
    rank = torch.max(cosine_similarity_mask_sorted_indices, dim=1, keepdim=True).values + 1

    # Take the reciprocal
    rr = 1 / rank
    
    # Get the mean
    return rr.sum() / cosine_similarity.shape[0]
    

# Implement mean reciprocal rank metric for evaluation

def mean_avg_precision_at_k(audio_embeddings, caption_embeddings, k=10):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
            - k: Scalar
    output: - mean_avg_precision_at_k: Scalar.
    
    """
    # The way this is done right now tests the model's ability to retrieve
    # the correct captions from audio
    ret = None
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = audio_embeddings @ caption_embeddings.T

    # Find unique audio embeddings
    unique_audio_embeddings = torch.unique(audio_embeddings, dim=0)

    # Find indices for each unique audio embedding
    unique_audio_embedding_indices = [torch.where(torch.all(audio_embeddings == unique_audio_embeddings[i], dim=1))[0] for i in range(unique_audio_embeddings.shape[0])]

    # Create zero like tensor same shape as cosine similarity
    cosine_similarity_mask = torch.zeros_like(cosine_similarity)
    
    # For each unique audio embedding, compute the combinations of indices taken 2 at a time
    for i in range(len(unique_audio_embedding_indices)):
        for j in range(len(unique_audio_embedding_indices[i])):
            for h in range(j, len(unique_audio_embedding_indices[i])):
                cosine_similarity_mask[unique_audio_embedding_indices[i][j], unique_audio_embedding_indices[i][h]] = 1
                cosine_similarity_mask[unique_audio_embedding_indices[i][h], unique_audio_embedding_indices[i][j]] = 1


    # Sort cosine similarity in descending order row wise and get the indices
    cosine_similarity_sorted, cosine_similarity_sorted_indices = torch.sort(cosine_similarity, dim=1, descending=True)

    # Sort cosine similarity mask row wise using the indices from above
    cosine_similarity_mask_sorted = cosine_similarity_mask[torch.arange(cosine_similarity_mask.shape[0]).unsqueeze(1), cosine_similarity_sorted_indices]

    # Sum over the first k columns of the sorted cosine similarity mask and divide by k
    ret = (cosine_similarity_mask_sorted[:, :k].sum(dim=1) / k).mean().item()

    return ret


def mean_recall_at_k(audio_embeddings, caption_embeddings, k=10):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
            - k: Scalar
    output: - recall_at_k: Scalar.
    
    """
    # The way this is done right now tests the model's ability to retrieve
    # the correct captions from audio
    ret = None

    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    # Each row in the cosine similarity matrix will be for a given audio embedding
    # With each column being a similarity score between that audio embedding and the caption embedding
    # at that index
    cosine_similarity = audio_embeddings @ caption_embeddings.T

    # Find unique audio embeddings
    unique_audio_embeddings = torch.unique(audio_embeddings, dim=0)
    # Find indices for each unique audio embedding
    unique_audio_embedding_indices = [torch.where(torch.all(audio_embeddings == unique_audio_embeddings[i], dim=1))[0] for i in range(unique_audio_embeddings.shape[0])]

    # Create zero like tensor same shape as cosine similarity
    cosine_similarity_mask = torch.zeros_like(cosine_similarity)
    
    # For each unique audio embedding, compute the combinations of indices taken 2 at a time
    for i in range(len(unique_audio_embedding_indices)):
        for j in range(len(unique_audio_embedding_indices[i])):
            for h in range(j, len(unique_audio_embedding_indices[i])):
                cosine_similarity_mask[unique_audio_embedding_indices[i][j], unique_audio_embedding_indices[i][h]] = 1
                cosine_similarity_mask[unique_audio_embedding_indices[i][h], unique_audio_embedding_indices[i][j]] = 1
    # Sort cosine similarity in descending order row wise and get the indices
    cosine_similarity_sorted, cosine_similarity_sorted_indices = torch.sort(cosine_similarity, dim=1, descending=True)
    # Sort cosine similarity mask row wise using the indices from above
    cosine_similarity_mask_sorted = cosine_similarity_mask[torch.arange(cosine_similarity_mask.shape[0]).unsqueeze(1), cosine_similarity_sorted_indices]
    # Sum over the first k columns of the sorted cosine similarity mask and divide by k
    ret = (cosine_similarity_mask_sorted[:, :k].sum(dim=1) / cosine_similarity_mask.sum(dim=1)).mean().item()

    return ret

if __name__ == "__main__":
    mrr = mean_reciprocal_rank(test_audio, test_captions)
    print(mrr)
    mapk = mean_avg_precision_at_k(test_audio, test_captions, 2)
    print(mapk)
    recallk = mean_recall_at_k(test_audio, test_captions, 3)
    print(recallk)