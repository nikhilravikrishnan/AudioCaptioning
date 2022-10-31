import torch
from torch.nn import functional as F

def load_pretrained_img_model(model, device, checkpoint_path):
    pretrained_model = model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model"])
    return pretrained_model


def mean_reciprocal_rank(model, audio_embeddings, caption_embeddings):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
    output: - mean_reciprocal_rank: Scalar.

    """
    ret = None
    
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
    caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    cosine_similarity = caption_embeddings @ audio_embeddings.T
    
    
    return ret

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

    return ret


def recall_at_k(model, audio_embeddings, caption_embeddings, k=10):
    """
    This function will implement the mean reciprocal rank function.
    input:  - model
            - video_embedding: Torch tensor ()
            - caption_embedding: Torch tensor ()
            - k: Scalar
    output: - recall_at_k: Scalar.
    
    """
    ret = None
    
    return ret

"""

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])

    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()

"""