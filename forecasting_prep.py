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
from data_loader_full import load_forecasting_data
# from model.classification import TSClassification
import sys

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
    data_obj = load_forecasting_data(config)
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    test_loader = data_obj["test_dataloader"]

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", 
        model_kwargs={
            'task_name': 'forecasting', #must specify the horizon
            'forecast_horizon': 512
        },
    )
    model.init()

    # Number of parameters in the encoder
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters: {num_params}")

    model = model.to(device).float()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    batch_count = 0

    for batch_idx, batch in enumerate(test_loader):
        # Get data
        
        seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
        future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
        # print('checking data shape', seen_data.shape, future_data.shape)
        ## prepare the data for moment model
        n_channels = seen_data.shape[2] 


        seen_data = seen_data.permute(0, 2, 1).to(device).float()
        future_data = future_data.permute(0, 2, 1).to(device).float()
        future_mask = future_mask.permute(0, 2, 1).to(device).float()

        # pad to 512 
        seen_data = torch.nn.functional.pad(seen_data, (0, 512 - seen_data.shape[2]), mode='constant', value=0)
        future_data = torch.nn.functional.pad(future_data, (0, 512 - future_data.shape[2]), mode='constant', value=0)
        future_mask = torch.nn.functional.pad(future_mask, (0, 512 - future_mask.shape[2]), mode='constant', value=0)

        # Store original dimensions
        batch_size, n_channels, seq_len = seen_data.shape
        # then reshape to the moment dimensions:
        # seen_data = seen_data.reshape(-1,1, seq_len)
        # future_data = future_data.reshape(-1,1, seq_len)
        # future_mask = future_mask.reshape(-1,1, seq_len)

        # load the moment model 
        output = model(x_enc=seen_data)
        pred_y = output.forecast
        # print('checking pred_y shape', pred_y.shape)

        # compute error data_to_predict & pred_y * mask_predicted_data

        # Replace your current MSE calculation with this
        if future_mask.sum() > 0:
            mse_loss = ((future_data - pred_y)**2) * future_mask
            mse_loss = mse_loss.sum() / future_mask.sum()

            mae_loss = (future_data - pred_y).abs()
            mae_loss = mae_loss * future_mask
            mae_loss = mae_loss.sum() / future_mask.sum()
            
            # Skip extreme values
            if mse_loss.item() < 1000 and not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                total_mse += mse_loss.item()
                total_mae += mae_loss.item()
                batch_count += 1
            else:
                print(f"Skipping batch {batch_idx} with extreme MSE: {mse_loss.item()}")
        else:
            print(f"Batch {batch_idx} has no prediction points (mask sum is zero)")

        print('checking mse loss', mse_loss.item())
        print('checking mae loss', mae_loss.item())
    print(total_mse)
    fina_mse = total_mse / batch_count
    fina_mae = total_mae / batch_count
    print('final mse', fina_mse)
    print('final mae', fina_mae)
