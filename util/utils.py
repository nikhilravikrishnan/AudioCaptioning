import torch

def load_pretrained_img_model(model, device, checkpoint_path):
    pretrained_model = model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model"])
    return pretrained_model