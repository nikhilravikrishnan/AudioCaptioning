import yaml
import argparse
import os
import sys
import torch.utils.data
import datetime
from models.clip import BaseClip, ViTClip, PANNClip
from torch.utils.data.dataloader import DataLoader
from dataloaders.clotho_dataloader import AudioCaptioningDataset, get_numpy_from_datadir
from util.utils import eval_model_embeddings
import torch.optim
import wandb

parser = argparse.ArgumentParser(description="Music caption retrieval project for Georgia Tech CS7643")
parser.add_argument("--config", default="configs/resnet.yaml")
parser.add_argument("--mode", default="train")

def set_syspath():
    sys.path.append(args.sys_path)
    sys.path.append(args.model_lib_path)
    print("Current sys.path settings:")
    print(sys.path)
    return

def train(get_metrics=False, eval_batch_size = 512, print_every_epoch = 1):
    """
    Make predictions on the dataset using the model specified using args.model on Mel-Spectrogram image representations
    of input audio.

    See the config_format.md file in the configs folder for more setting details.

    Parameters
    ---
    get_metrics: boolean
        Whether to get metrics (MRR, Recall/Precision @ 5) each epoch of training

    """
    config = {"lr": args.lr, "batch_size":args.batch_size, "seed":args.random_seed}
    
    wandb.init(project=args.model + "-F22", entity="deep-learning-f22",
              config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device=device)
    
    # Setting the random seed for reproducibility if needed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
    
    print(f"Running on device: {device}")

    # Settings to save the model
    
    model_dir = args.save_dir
    if args.random_seed is not None:
        model_dir += f"seed_{args.random_seed}_{datetime.date.today():%Y%m%d%H%M%S}"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(model_dir)
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(model_dir)

    epochs = args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr =args.lr, weight_decay=0.)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1.0, factor=0.8)

    # Create train and validation dataloaders
    print("Creating dataloaders...")
    data_train = get_numpy_from_datadir(args.data_dir, 'train/val')

    if args.model == 'PANN':
      train_dataset = AudioCaptioningDataset(data_train['train_spectrograms'], data_train['train_captions'], augment = True)
      val_dataset = AudioCaptioningDataset(data_train['val_spectrograms'], data_train['val_captions'])
    else:
      train_dataset = AudioCaptioningDataset(data_train['train_spectrograms'], data_train['train_captions'], augment = True, multichannel = True)
      val_dataset = AudioCaptioningDataset(data_train['val_spectrograms'], data_train['val_captions'], multichannel = True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    train_dataloader_metrics = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=True)
    val_dataloader_metrics = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True)

    min_val_loss = float("inf")

    # Training and Validation
    print("Starting training!")
    
    for e in range(epochs):
        start = datetime.datetime.now()
        print(f"Beginning epoch {e+1} at {start}.")
        # Training loss
        train_total_loss = 0
        # Set model in train mode
        model.train()
        # Loop over train data loader
        for (idx,batch) in enumerate(train_dataloader):
            if idx % 10 == 0:
                batch_time = datetime.datetime.now()
                print(f"Training batch {idx+1} processed at: {batch_time}")

            # Send to device
            batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            # Forward pass
            batch_loss, audio_encoders, text_encoders = model.forward(batch)
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            # Optimizer step
            optimizer.step()
            lr_scheduler.step(metrics=batch_loss)
            # Accumulating gradient
            train_total_loss += batch_loss.item()
        
        # Logs to wandb every iteration
        print('Training Loss:', train_total_loss/len(train_dataloader))
        wandb.log({'Training Loss': train_total_loss/len(train_dataloader)})
        # Saving model
        save_filename = model_dir + f"/model_{e+1}.pth"
        if get_metrics == True:
            # Training metrics evaluation       
            train_metrics = evaluate(model, train_dataloader_metrics, "train")
            # Logs to wandb
            wandb.log({'train MRR': train_metrics["train_MRR"], 
                       'train MAP@K': train_metrics['train_MAP@K'], 
                       'train R@K': train_metrics['train_R@K']})
            # Print metrics to terminal
            if (e+1) % print_every_epoch == 0 or ((e+1) == epochs):
                    print ('Training Metrics. Epoch: [{}/{}], MRR: {:.4f}, MAP@K: {:.4f}, R@K: {:.4f}' 
                          .format(e + 1, epochs, 
                                  train_metrics["train_MRR"], 
                                  train_metrics["train_MAP@K"],
                                  train_metrics["train_R@K"]))
            # Saving metrics for train split
            train_metrics_fp = model_dir + "/train_metrics.txt"
            print(f"Saving final training metrics to: {train_metrics_fp}")
            with open(train_metrics_fp, "w+") as f:
                for k in train_metrics.keys():
                    f.write(k + ": " + f"{train_metrics[k]}\n")

        # Set model in evaluation mode
        model.eval()
        # Total val loss
        val_total_loss = 0
        # Validation loop
        for (idx,batch) in enumerate(val_dataloader):
            # Printing eval batch 
            if idx % 10 == 0:
                batch_time = datetime.datetime.now()
                print(f"Eval batch {idx+1} processed at: {batch_time}")
            # Forward pass and loss calculation
            batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            batch_loss, _, __ = model.forward(batch)
            val_total_loss += batch_loss.item()
        # Logging the loss
        print('Validation Loss:', val_total_loss/len(val_dataloader))
        wandb.log({'Validation Loss': val_total_loss/len(val_dataloader)})  

        if get_metrics == True:
            # Val metrics evaluation       
            val_metrics = evaluate(model, val_dataloader_metrics, "val")
            # Logs to wandb
            wandb.log({'val MRR': val_metrics["val_MRR"], 
                       'val MAP@K': val_metrics['val_MAP@K'], 
                       'val R@K': val_metrics['val_R@K']})
            # Print metrics to terminal
            if (e+1) % print_every_epoch == 0 or ((e+1) == epochs):
              print ('Validation Metrics. Epoch: [{}/{}], MRR: {:.4f}, MAP@K: {:.4f}, R@K: {:.4f}' 
                    .format(e + 1, epochs, 
                            val_metrics["val_MRR"], 
                            val_metrics["val_MAP@K"],
                            val_metrics["val_R@K"]))
            # Saving metrics for val split
            val_metrics_fp = model_dir + "/val_metrics.txt"
            print(f"Saving final validation metrics to: {val_metrics_fp}")
            with open(val_metrics_fp, "w+") as f:
                for k in val_metrics.keys():
                    f.write(k + ": " + f"{val_metrics[k]}\n")

        # If new loss is better than previous best, saves the model
        if val_total_loss < min_val_loss:
            print("Saving...")
            torch.save(model.state_dict(), save_filename)
            print('Saved as %s' % save_filename)  
            min_val_loss = val_total_loss

    return

def evaluate(model, dataloader, stage = 'train'):
    # Set the model in eval mode
    model.eval()

    # Use the GPU if we can
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device {device} for model evaluation.")    
    metrics = eval_model_embeddings(model, device, dataloader, ["MRR", "MAP@K", "R@K"], k=10)

    for k in metrics.keys():
      metrics[stage+'_'+k] = metrics.pop(k)

    if stage != 'test':
      wandb.log({stage+' MRR': metrics[stage+"_MRR"], 
                  stage+' MAP@K': metrics[stage+'_MAP@K'], 
                  stage+' R@K': metrics[stage+'_R@K']})
    
    return metrics

def load_model(model_type, device, state_dict=None):
    """
    Load a specified model with newly initialized weights or with a model
    state loaded from a state dict at the specified file path.
    """

    # Creates a version of a Clip depending on model type
    model = None
    if model_type == "ResNet":
        model = BaseClip(device=device, fine_tune=args.fine_tune)
    elif model_type == "ViT":
        model = ViTClip(device=device, fine_tune=args.fine_tune)
    elif model_type == "PANN":
        model = PANNClip(device=device, fine_tune=args.fine_tune)
    
    # Loads the state dict for the model if exists
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
    print(args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    # Setting the sys.path variable so we can find our models' Python modules
    set_syspath()

    # Make predictions using the appropriate method for the selected model
    if args.mode == "train":
      # Starts training
      train(get_metrics=args.get_metrics, eval_batch_size = 512, print_every_epoch = 1)
    
    if args.mode == "test":
      # Eval batch_size
      eval_batch_size = 32

      # Checks device
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # Loads model
      model = load_model(args.model, device, args.checkpoint_path)

      # Loads data set and data loader
      data_test = get_numpy_from_datadir(args.data_dir, 'test')
      if args.model == 'PANN':
        test_dataset = AudioCaptioningDataset(data_test['test_spectrograms'], data_test['test_captions'])
      else:
        test_dataset = AudioCaptioningDataset(data_test['test_spectrograms'], data_test['test_captions'], multichannel = True)
      test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

      # Evaluates metrics for data loader
      test_metrics = evaluate(model, test_dataloader, stage="test")
      test_metrics_fp = args.save_dir + "/{args.model}_test_metrics.txt"

      # Prints metrics:
      print ('Test Metrics. MRR: {:.4f}, MAP@K: {:.4f}, R@K: {:.4f}' 
            .format(test_metrics["test_MRR"], 
                    test_metrics["test_MAP@K"],
                    test_metrics["test_R@K"]))

      # Writing metrics
      with open(test_metrics_fp, "w+") as f:
          for k in test_metrics.keys():
              f.write(k + ": " + f"{test_metrics[k]}\n")
    
    return
        

if __name__ == "__main__":
    main()    