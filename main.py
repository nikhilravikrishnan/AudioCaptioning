import yaml
import argparse
import os
import sys
import torch.utils.data
import datetime

parser = argparse.ArgumentParser(description="Music caption retrieval project for Georgia Tech CS7643")
parser.add_argument("--config", default="/home/jupyter/music/configs/resnet.yaml")
parser.add_argument("--mode", default="train")

def set_syspath():
    sys.path.append(args.sys_path)
    sys.path.append(args.model_lib_path)
    print("Current sys.path settings:")
    print(sys.path)
    return

def train(get_metrics=False, fine_tune=False):
    """
    Make predictions on the dataset using the model specified using args.model on Mel-Spectrogram image representations
    of input audio.

    See the config_format.md file in the configs folder for more setting details.

    Parameters
    ---
    get_metrics: boolean
        Whether to get metrics (MRR, Recall/Precision @ 5) each epoch of training

    """
    from models.clip import BaseClip, ViTClip
    import models.trainable_params
    from torch.utils.data.dataloader import DataLoader
    from dataloaders.clotho_dataloader import AudioCaptioningDataset
    import torch.optim
    import wandb

    wandb.init(project=args.model + "-F22", entity="deep-learning-f22")

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
    optimizer = None

    if fine_tune == False:
        optimizer = torch.optim.Adam(model.parameters(), lr =1e-3, weight_decay=0.)
    else:
        if args.model == "ViT":
            model_params = models.trainable_params.get_trainable_vit_params(model, args.num_trainable_layers)
        optimizer = torch.optim.Adam(model_params, lr=1e-4, weight_decay=0.01)
    
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
        wandb.log({'Training Loss': train_total_loss/len(train_dataloader)})
        print('Epoch:', e)

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
        wandb.log({'Validation Loss': val_total_loss/len(val_dataloader)})  

        if val_total_loss < min_val_loss:
            print("Saving...")
            torch.save(model.state_dict(), save_filename)
            print('Saved as %s' % save_filename)  
            min_val_loss = val_total_loss

            if get_metrics == True:
                train_metrics = evaluate(model, "train")
                print(f"Epoch {e} training metrics: ")
                print(train_metrics)
    
    metrics = evaluate(model, "train")
    metrics_fp = model_dir + "/train_metrics.txt"
    print(f"Saving final training metrics to: {metrics_fp}")

    with open(metrics_fp, "w+") as f:
        for k in metrics.keys:
            f.write(k + ": " + f"{metrics[k]}\n")

    return

def evaluate(model, mode="eval", mean=True):
    from torch.utils.data.dataloader import DataLoader
    from dataloaders.clotho_dataloader import AudioCaptioningDataset
    from util.utils import eval_model_embeddings
    
    # Use the GPU if we can
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device {device} for model evaluation.")
    
    dataset = AudioCaptioningDataset(data_dir = args.data_dir, split=args.split)

    if mode == "eval":
        all_batch_metrics = {"MRR":[], "MAP@K":[], "R@K":[]}
        dataset_ind = [i for i in range(0, len(dataset), 100)] + [len(dataset)]
        
        for i in range(len(dataset_ind)):
            if i == 0:
                continue
            dataset_sub = torch.utils.data.Subset(dataset, range(dataset_ind[i-1], dataset_ind[i]))

            dataloader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)
            
            print(f"Calculating metrics for items {dataset_ind[i-1]} - {dataset_ind[i]}...")
            batch_metrics = eval_model_embeddings(model, device, dataloader, ["MRR", "MAP@K", "R@K"], k=5)
            for k in batch_metrics.keys():
                all_batch_metrics[k].append(batch_metrics[k])
    
    if mode == "train":
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataset_ind = [i for i in range(0, len(train_dataset), 100)] + [len(train_dataset)]
        val_dataset_ind = [i for i in range(0, len(val_dataset), 100)] + [len(val_dataset)]

        all_batch_metrics = {"train_MRR":[], "train_MAP@K":[], "train_R@K":[], 
                            "val_MRR":[], "val_MAP@K":[], "val_R@K":[]}

        print("Calculating training set metrics...")
        for i in range(len(train_dataset_ind)):
            if i == 0:
                continue
            print(f"Calculating metrics for items {train_dataset_ind[i-1]} - {train_dataset_ind[i]}...")
            dataset_sub = torch.utils.data.Subset(train_dataset, range(train_dataset_ind[i-1], train_dataset_ind[i]))

            dataloader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)
            
            batch_metrics = eval_model_embeddings(model, device, dataloader, ["MRR", "MAP@K", "R@K"], k=5)
            for k in batch_metrics.keys():
                all_batch_metrics["train_"+k].append(batch_metrics[k])

        print("Calculating validation set metrics...")
        for i in range(len(val_dataset_ind)):
            if i == 0:
                continue
            print(f"Calculating metrics for items {val_dataset_ind[i-1]} - {val_dataset_ind[i]}...")
            dataset_sub = torch.utils.data.Subset(val_dataset, range(val_dataset_ind[i-1], val_dataset_ind[i]))

            dataloader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)
            
            batch_metrics = eval_model_embeddings(model, device, dataloader, ["MRR", "MAP@K", "R@K"], k=5)
            for k in batch_metrics.keys():
                all_batch_metrics["val_"+k].append(batch_metrics[k])

    if mean == True:
        print(all_batch_metrics)
        # Get the mean of all batches for each metric rather than the individual results of each batch
        for k in all_batch_metrics.keys():
            metric_mean = sum(all_batch_metrics[k]) / len(all_batch_metrics[k])
            all_batch_metrics[k] = metric_mean

    return all_batch_metrics

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
        if device.type != "cuda":
            model.load_state_dict(torch.load(state_dict, map_location=torch.device("cpu")))
        else:
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

    args.mode = "eval"
    
    # Setting the sys.path variable so we can find our models' Python modules
    set_syspath()

    # Make predictions using the appropriate method for the selected model
    if args.mode == "train":
        train(get_metrics=args.get_metrics, fine_tune=args.fine_tune)
    
    if args.mode == "eval":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model(device, args.model_fp)

        metrics = evaluate(model, mode="eval")

        metrics_fp = args.save_dir + "/eval_metrics.txt"

        with open(metrics_fp, "w+") as f:
            for k in metrics.keys():
                f.write(k + ": " + f"{metrics[k]}\n")
    
    return
        

if __name__ == "__main__":
    main()