import torch
import torch.nn as nn

def get_trainable_vit_params(model, num_layers=2):
    max_num_layers = 11
    training_params = [
        {"params": model.audio_encoder.heads.parameters(), "lr": 5e-3},
        {"params": model.audio_encoder.encoder.ln.parameters(), "lr": 5e-3}
    ]
    all_params = {name: parameter for name, parameter in model.named_parameters()}
    
    # Step backward through the layers - fine tune the last layers first.
    # Increase the number of layers to train earlier layers.
    for l in range(max_num_layers, max_num_layers-num_layers, -1):
        lr = 1 * (10 ** (l - 14)) # The learning rate decreases as we get to earlier layers - probably will need tweaking
        for k in all_params.keys():
            if str(l) in k and "audio" in k:
                training_params.append({"params": all_params[k], "lr": lr})
    return training_params