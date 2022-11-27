import yaml
import argparse
import os
import sys
import torch.utils.data
import datetime

parser = argparse.ArgumentParser(description="Music caption retrieval project for Georgia Tech CS7643")
parser.add_argument("--config", default="/home/jupyter/music/configs/resnet.yaml")

def set_syspath():
    sys.path.append(args.sys_path)
    sys.path.append(args.model_lib_path)
    print("Current sys.path settings:")
    print(sys.path)
    return

def train():
    """
    Make predictions on the dataset using a ResNet-50 model on Mel-Spectrogram image representations
    of input audio.

    The embedding for each image is made up of weights retrieved from the avgpool layer for that image after
    a forward pass through the network. This is done using the "create_feature_extractor" method from torchvision,
    but can also be done by setting a forward hook.

    (See this helpful post:
    https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model)

    """
    from models.clip import BaseClip, ViTClip
    from torch.utils.data.dataloader import DataLoader
    from dataloaders.clotho_dataloader import AudioCaptioningDataset
    import torch.optim 
    from tensorboard_logger import configure, log_value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device=device)
    
    # Setting the random seed for reproducibility if needed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
    
    using_cuda = torch.cuda.is_available()
    print(f"Running on CUDA: {using_cuda}")

    # Settings to save the model
    
    model_dir = args.save_dir
    if args.random_seed is not None:
        model_dir += f"seed_{args.random_seed}"

    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr =1e-3, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1.0, factor=0.8)


    dataset = AudioCaptioningDataset(data_dir = args.data_dir, split=args.split)

    # Create train and validation dataloaders
    print("Creating dataloaders...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    min_val_loss = float("inf")

    configure(model_dir + '/runs/', flush_secs=5)

    # Training and Validation
    print("Starting training!")
    
    for e in range(epochs):
        start = datetime.datetime.now()
        print(f"Beginning epoch {e} at {start}.")
        # Training
        train_total_loss = 0
        model.train()
        for (idx,batch) in enumerate(train_dataloader):
            if idx % 10 == 0:
                batch_time = datetime.datetime.now()
                print(f"Training batch {idx} processed at: {batch_time}")

            # Send to device
            batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            
            batch_loss, audio_encoders, text_encoders = model.forward(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step(metrics=batch_loss)
            train_total_loss += batch_loss.item()
        
        
        print('Training Loss:', train_total_loss/len(train_dataloader))
        print('Epoch:', e)

        log_value('train_loss', train_total_loss/len(train_dataloader), e)
        log_value('learning_rate', optimizer.param_groups[0]['lr'], e)

        save_filename = model_dir + f"/model_{e}.pth"

        model.eval()
        val_total_loss = 0
        
        # Validation
        for (idx,batch) in enumerate(val_dataloader):
            if idx % 10 == 0:
                batch_time = datetime.datetime.now()
                print(f"Eval batch {idx} processed at: {batch_time}")
            
            batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            batch_loss, _, __ = model.forward(batch)
            val_total_loss += batch_loss.item()

        print('Validation Loss:', val_total_loss/len(val_dataloader))  

        log_value('val_loss', val_total_loss/len(val_dataloader), e)
        log_value('learning_rate', optimizer.param_groups[0]['lr'], e)

        if val_total_loss < min_val_loss:
            print("Saving...")
            torch.save(model.state_dict(), save_filename)
            print('Saved as %s' % save_filename)  
            min_val_loss = val_total_loss

    metrics = evaluate(model, "train")
    metrics_fp = model_dir + "/train_metrics.txt"

    with open(metrics_fp, "w+") as f:
        for k in metrics.keys:
            f.write(k + ": " + f"{metrics[k]}\n")

    return

def evaluate(model, mode="eval"):
    from torch.utils.data.dataloader import DataLoader
    from dataloaders.clotho_dataloader import AudioCaptioningDataset
    from util.utils import eval_model_embeddings
    
    # Use the GPU if we can
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device {device} for model evaluation.")
    
    dataset = AudioCaptioningDataset(data_dir = args.data_dir, split=args.split)
    metrics = {}

    if mode == "eval":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
        metrics["mrr"] = eval_model_embeddings(model, dataloader, "MRR")
        metrics["map@5"] = eval_model_embeddings(model, dataloader, "MAP@K", k=5)
        metrics["r@5"] = eval_model_embeddings(model, dataloader, "R@K", k=5)

    if mode == "train":
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        metrics["train_mrr"] = eval_model_embeddings(model, train_dataloader, "MRR")
        metrics["train_map@5"] = eval_model_embeddings(model, train_dataloader, "MAP@K", k=5)
        metrics["train_r@5"] = eval_model_embeddings(model, train_dataloader, "R@K", k=5)

        metrics["val_mrr"] = eval_model_embeddings(model, val_dataloader, "MRR")
        metrics["val_map@5"] = eval_model_embeddings(model, val_dataloader, "MAP@K", k=5)
        metrics["val_r@5"] = eval_model_embeddings(model, val_dataloader, "R@K", k=5)

    return metrics

def load_model(device, state_dict=None):
    """
    Load a specified model with newly initialized weights or with a model
    state loaded from a state dict at the specified file path.
    """
    from models.clip import BaseClip, ViTClip

    # Use the GPU if we can

    model = None

    if args.model == "ResNet":
        model = BaseClip(device=device)
    elif args.model == "ViT":
        model = ViTClip(device=device)
    else:
        raise NotImplemented

    if state_dict != None:
        model.load_state_dict(torch.load(state_dict))

    return model


def main():
    # Use a config file to make sure we perform the correct experimental setup
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    # Setting the sys.path variable so we can find our models' Python modules
    set_syspath()

    # Make predictions using the appropriate method for the selected model
    if args.mode == "train":
        train()
    
    if args.mode == "eval":

        model = load_model(args.model_fp)

        metrics = evaluate(model, mode="eval")

        metrics_fp = args.save_dir + "/eval_metrics.txt"

        with open(metrics_fp, "w+") as f:
            for k in metrics.keys:
                f.write(k + ": " + f"{metrics[k]}\n")
    
    return
        

if __name__ == "__main__":
    main()