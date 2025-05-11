import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from logger import setup_logger
from config import Config
from data_loader_full import load_interpolation_data
# from model.classification import TSClassification
import sys

#### moment packages
from momentfm.utils.utils import control_randomness
control_randomness(seed=0) # Set random seeds for PyTorch, Numpy etc.

from momentfm import MOMENTPipeline
from momentfm.data.informer_dataset import InformerDataset
from torch.utils.data import DataLoader
from momentfm.utils.masking import Masking
from tqdm import tqdm
from momentfm.utils.forecasting_metrics import mse, mae

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def subsample_timepoints(data, time_steps, mask, config):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * config.percentage_tp_to_sample) ## increase the data will get maskes less, while decrea, the data will get masked
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Time Series Classification")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    args = parser.parse_args()
    
    # Load config
    config = Config()
    if args.config and os.path.exists(args.config):
        config.load(args.config)
    
    # Additional config for classification task
    config.freeze_base = False  # Whether to freeze pretrained model parameters
    config.cls_lr = 0.001  # Learning rate for classification task
    config.cls_epochs = 100  # Number of epochs
    config.early_stopping_patience = 15
    args.pretrained = config.pretrained_path
    
    # Setup logger
    logger, exp_dir = setup_logger(config.log_dir, "classification_" + config.experiment_name)
    config.save(os.path.join(exp_dir, "config.json"))
    
    ##### Load the task-specific data #####
    data_obj = load_interpolation_data(config)
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    test_loader = data_obj["test_dataloader"]

    ## load the moment model 

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", 
        model_kwargs={"task_name": "reconstruction"},
    )
    model.init()
    # Number of parameters in the encoder
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters: {num_params}")
    
    model = model.to(device).float()
    
    # our data 
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        # Get data
        data, time_steps, mask = batch['data'], batch['time_steps'], batch['mask']
        # print('checking data shape', data.shape)
        original_data = data.clone()
        original_mask = mask.clone()

        #### perform the subsampling ####
        subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                data.clone(), time_steps.clone(), mask.clone(), config
                )   
        
        interp_mask = original_mask - subsampled_mask  # Points to interpolate
        # check whether the interp_mask is 0 or not
        if interp_mask.sum() == 0:
            print('no points to interpolate')
            continue

        # print('checking subsampled data shape', subsampled_data.shape,subsampled_mask.shape)

        n_channels = data.shape[2] 

        ## first, let's permute and pad the data to the same lenght max_len, from [batch_size, seq_len, n_channels] to [batch_size, n_channels, max_len]
        original_data = original_data.permute(0, 2, 1)
        subsampled_data = subsampled_data.permute(0, 2, 1)
        original_mask = original_mask.permute(0, 2, 1)
        subsampled_mask = subsampled_mask.permute(0, 2, 1)
        interp_mask = interp_mask.permute(0, 2, 1)
        
        original_data = nn.functional.pad(original_data, (0, config.max_len - original_data.size(2)), value=0)
        subsampled_data = nn.functional.pad(subsampled_data, (0, config.max_len - subsampled_data.size(2)), value=0)
        original_mask = nn.functional.pad(original_mask, (0, config.max_len - original_mask.size(2)), value=0)
        subsampled_mask = nn.functional.pad(subsampled_mask, (0, config.max_len - subsampled_mask.size(2)), value=0)
        interp_mask = nn.functional.pad(interp_mask, (0, config.max_len - interp_mask.size(2)), value=0)

        # first, we need to reshape data from [batch_size, seq_len, n_channels] to [batcg_size, n_cannels, seq_len]
        # and then to [batch_size * n_channels, 1, seq_len]
        # add pad them to the same length 512 for both data and mask 
       
        # Store original dimensions
        batch_size, n_channels, seq_len = original_data.shape

        # Then permute and reshape with the correct dimensions
        original_data = original_data.reshape(-1, 1, seq_len)
        subsampled_data = subsampled_data.reshape(-1, 1, seq_len)
        batch_mask = original_mask.reshape(-1, seq_len)
        subsampled_mask = subsampled_mask.reshape(-1, seq_len)
        interp_mask = interp_mask.reshape(-1, seq_len)

        original_data = original_data.float()
        subsampled_data = subsampled_data.float()

        output = model(x_enc=subsampled_data)  ## use this, rather than put the mask in the model..
        reconstruction = output.reconstruction.detach()

        reconstruction = reconstruction.float()      

        reconstruction = reconstruction.reshape((-1, n_channels, seq_len))
        original_data = original_data.reshape((-1, n_channels, seq_len))
        interp_mask = interp_mask.reshape((-1, n_channels, seq_len))
        mse_loss = (original_data - reconstruction) ** 2
        mse_loss = mse_loss * interp_mask
        mse_loss = mse_loss.sum() / interp_mask.sum()
        # print('checking mse loss', mse_loss.item())
        # print(f"Original data stats: min={original_data.min()}, max={original_data.max()}, has_nan={torch.isnan(original_data).any()}")
        # print(f"Interp mask sum: {interp_mask.sum()}")

        mae_loss = (original_data - reconstruction).abs()
        mae_loss = mae_loss * interp_mask
        mae_loss = mae_loss.sum() / interp_mask.sum()

        total_mse += mse_loss.item() * batch_size
        total_mae += mae_loss.item() * batch_size
        total_samples += batch_size

    fina_mse = total_mse / total_samples
    fina_mae = total_mae / total_samples
    print('final mse', fina_mse)
    print('final mae', fina_mae)