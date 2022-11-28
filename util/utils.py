import torch
import torch.nn.functional as F

def load_pretrained_img_model(model, device, checkpoint_path):
    pretrained_model = model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model"])
    return pretrained_model

test_audio = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_captions = torch.Tensor([[1,2,3], [4,5,6], [1,2,3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])

def eval_model_embeddings(model, dataLoader, metric_name: list, **kwargs):
    """
    Obtain the evaluation metric of the specified type from the given model
    input:  - model: CLIP model
            - metric_name: a list containing 1 or more of the following: ['MRR', 'MAP@K', 'R@K']
    output: - metric specified
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio_embedding_list = []
    text_embedding_list = []

    print("Generating embeddings...")
    for (idx, batch) in enumerate(dataLoader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        _, audio_encoders, text_encoders = model.forward(batch)
        audio_embedding_list.append(audio_encoders)
        text_embedding_list.append(text_encoders)

    # Concatenate all the embeddings
    audio_embeddings = torch.cat(audio_embedding_list, dim = 0)
    text_embeddings = torch.cat(text_embedding_list, dim = 0)

    metrics = {}

    if 'MRR' in metric_name:
        print("Calculating MRR...")
        metrics["MRR"] = mean_reciprocal_rank(audio_embeddings, text_embeddings)
    
    if 'MAP@K' in metric_name:
        if 'k' not in kwargs:
            raise ValueError("Needs K parameter.")
        print("Calculating MAP@K...")
        metrics["MAP@K"] = mean_avg_precision_at_k(audio_embeddings, text_embeddings, k = kwargs['k'])

    if 'R@K' in metric_name:
        if 'k' not in kwargs:
            raise ValueError("Needs K parameter.")
        print("Calculating R@K...")
        metrics["R@K"] = mean_recall_at_k(audio_embeddings, text_embeddings, k = kwargs['k'])

    return metrics

def evaluate_precalcuated_embeddings(audio_embeddings, text_embeddings, metric_name: list, num_batches=8, **kwargs):
    metrics = {"MRR":[], "MAP@K":[], "R@K":[]}
    
    # Start and stop indicies to slice the tensors into batches that fit into memory
    batches = [i for i in range(0, audio_embeddings.shape[0], int(audio_embeddings.shape[0]/num_batches))] + [audio_embeddings.shape[0]]

    # Calculate each of the requested metrics for each batch
    for b in range(len(batches)):
        start = batches[b]
        end = batches[b+1]

        if 'MRR' in metric_name:
            print("Calculating MRR...")
            metrics["MRR"].append(mean_reciprocal_rank(audio_embeddings[start:end, :], text_embeddings[start:end, :]))
    
        if 'MAP@K' in metric_name:
            if 'k' not in kwargs:
                raise ValueError("Needs K parameter.")
            print("Calculating MAP@K...")
            metrics["MAP@K"].append(mean_avg_precision_at_k(audio_embeddings[start:end, :], text_embeddings[start:end, :], k = kwargs['k']))

        if 'R@K' in metric_name:
            if 'k' not in kwargs:
                raise ValueError("Needs K parameter.")
            print("Calculating R@K...")
            metrics["R@K"].append(mean_recall_at_k(audio_embeddings[start:end, :], text_embeddings[start:end, :], k = kwargs['k']))

    # Return them as a list - we can average them later if desired
    return metrics

def mean_reciprocal_rank(audio_embeddings, caption_embeddings):
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

    return ((1/(torch.argmax(cosine_similarity, dim=1)+1)).sum() / cosine_similarity.shape[0]).item()

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
    ret = None
    
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
    ret = (cosine_similarity_mask_sorted[:, :k].sum(dim=1) / cosine_similarity_mask.sum(dim=1)).mean().item()

    return ret

if __name__ == "__main__":
    mrr = mean_reciprocal_rank(test_audio, test_captions)
    print(mrr)
    mapk = mean_avg_precision_at_k(test_audio, test_captions, 2)
    print(mapk)
    recallk = mean_recall_at_k(test_audio, test_captions, 3)
    print(recallk)