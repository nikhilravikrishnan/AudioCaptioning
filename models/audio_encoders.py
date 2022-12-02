import torch
import torch.nn as nn
import torch.functional as F
import torchvision

from torchvision.models.feature_extraction import create_feature_extractor


def get_audio_feature_vector(model: nn.Module, img_tensors: torch.Tensor, layer_dict):
    """
    Get feature vectors for the given audio (passed in as a spectrogram image) by returning the output
    of the selected model at an intermediate layer.

    For more info on create_feature_extractor, see the Torchvision documentation here:
    http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html

    Arguments
    -----
    model - A PyTorch nn.Module object
    img_tensor - A batch of images stored as PyTorch tensors with dimensions (n, c, h, w)
    layer_dict - A dictionary containing information about the desired intermediate layer output
        Ex. {'avgpool': 'features'} to obtain the weights at ResNet's avgpool layer stored with the key "features"

    Outputs
    -----
    feat_vec - A list of feature vectors for each of the requested layers as tuples in [("name", tensor)] format

    """
    features = []
    model = create_feature_extractor(model, layer_dict)
    img_tensors = img_tensors.unsqueeze(0)
    model = model.double()
    out = model(img_tensors)
    features = out["features"]
    features = torch.reshape(features, (features.size()[0], features.size()[1]))
    return features

def get_vit_feature_vector(model: nn.Module, device, img_tensors: torch.Tensor, layer_dict={"encoder.layers.encoder_layer_11.mlp":"features"}):
    """
    Get feature vectors for the given audio (passed in as a spectrogram image) by returning the output
    of the selected model at an intermediate layer.

    For more info on create_feature_extractor, see the Torchvision documentation here:
    http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html

    Arguments
    -----
    model - A PyTorch nn.Module object
    img_tensor - A batch of images stored as PyTorch tensors with dimensions (n, c, h, w)
    layer_dict - A dictionary containing information about the desired intermediate layer output
        Ex. {'avgpool': 'features'} to obtain the weights at ResNet's avgpool layer stored with the key "features"

    Outputs
    -----
    feat_vec - A list of feature vectors for each of the requested layers as tuples in [("name", tensor)] format

    """
    features = None
    model = create_feature_extractor(model, layer_dict).to(device)
    img_tensors = img_tensors.unsqueeze(0).to(device)
    model = model.double()
    out = model(img_tensors)
    features = out["features"]
    features = features[:, 0, :]
    return features