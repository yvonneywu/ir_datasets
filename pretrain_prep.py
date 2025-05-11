
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
from data_loader import load_data
# from model.classification import TSClassification
import sys
import random
from model.gmae_vnew import gMAE
from data_processing.mask_curation import CLDataCollator

# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Device count: {torch.cuda.device_count()}")
# if torch.cuda.is_available():
#     print(f"Current device: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name()}")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, epoch, logger, config):
    model.train()
    total_loss = 0
    start_time = time.time()
    # num_batches = len(dataloader)
    total_latent_loss = 0
    total_signal_loss = 0
    train_cl_collator = CLDataCollator(max_len=config.max_len, config=config)
    num_batches = 0

    for batch in train_iterator:
        # Get data
        dataset_idx = batch["dataset_idx"]
        # print('dataset_idx', dataset_idx)
        dataset_info = datasets_info[dataset_idx]
        # print('dataset_info', dataset_info)
        input_dim = dataset_info["input_dim"]
        # print('input_dim', input_dim)
        data, time_steps, mask= batch['data'], batch['time_steps'], batch['mask']
        # print('checking data shape', data.shape, time_steps.shape)

        #### modeling #### 
        # 1. artifically create the mask in the raw signal

         # Zero gradients
        optimizer.zero_grad()
        
        if config.mask_ori:
            value_batch, time_batch, mask_batch = train_cl_collator(batch)
            value_batch = value_batch.to(device)
            time_batch = time_batch.to(device)
            mask_batch = mask_batch.to(device)

            output = model(value_batch, time_batch, mask_batch)
        else:
            output = model(data, time_steps, mask)
        
        loss = output['loss']
        latent_loss = output['latent_loss']
        signal_loss = output['signal_loss']
        print('checking loss', loss, latent_loss, signal_loss)

        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        total_latent_loss += latent_loss.item()
        total_signal_loss += signal_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_latent_loss = total_latent_loss / num_batches
    avg_signal_loss = total_signal_loss / num_batches
    epoch_time = time.time() - start_time
    logger.info(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f} Avg latent loss: {avg_latent_loss:.6f} AVg signal loss: {avg_signal_loss:.6f} '
                f'Time: {epoch_time:.2f}s')
    return avg_loss

def validate(model, dataloader, logger, config):
    model.eval()
    total_loss = 0
    # num_batches = len(dataloader)
    train_cl_collator = CLDataCollator(max_len=config.max_len, config=config)
    num_batches = 0

    with torch.no_grad():
        for batch in train_iterator:
        # Get data
            dataset_idx = batch["dataset_idx"]
            # print('dataset_idx', dataset_idx)
            dataset_info = datasets_info[dataset_idx]
            # print('dataset_info', dataset_info)
            input_dim = dataset_info["input_dim"]
            # print('input_dim', input_dim)
            data, time_steps, mask= batch['data'], batch['time_steps'], batch['mask']
            # print('checking data shape', data.shape, time_steps.shape)

            #### modeling #### 
            # 1. artifically create the mask in the raw signal

            # Zero gradients
            optimizer.zero_grad()
            
            if config.mask_ori:
                value_batch, time_batch, mask_batch = train_cl_collator(batch)
                value_batch = value_batch.to(device)
                time_batch = time_batch.to(device)
                mask_batch = mask_batch.to(device)

                output = model(value_batch, time_batch, mask_batch)
            else:
                output = model(data, time_steps, mask)

            loss = output['loss']
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f'====> Validation loss: {avg_loss:.6f}')
    return avg_loss




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
    
    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    # Setup logger
    logger, exp_dir = setup_logger(config.log_dir, "pretrain_" + config.experiment_name)
    config.save(os.path.join(exp_dir, "config.json"))
    logger.info(f"Configuration saved to {exp_dir}")
    
    ##### Load the pretrain data from mimic and physionet #####
    data_obj = load_data(config) ## we can use the interpolation as the base
    datasets_info = data_obj["datasets_info"]
    train_iterator = data_obj["train_dataloader"]
    val_iterator = data_obj["val_dataloader"]

    # for batch in train_iterator:
    #     # Get data
    #     dataset_idx = batch["dataset_idx"]
    #     print('dataset_idx', dataset_idx)
    #     dataset_info = datasets_info[dataset_idx]
    #     print('dataset_info', dataset_info)
    #     input_dim = dataset_info["input_dim"]
    #     print('input_dim', input_dim)
    #     data, time_steps, mask= batch['data'], batch['time_steps'], batch['mask']
    #     print('checking data shape', data.shape, time_steps.shape)

    # sys.exit()

    #### define the model #### 
    model = gMAE(config).to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=5, verbose=True)

    # Training loop
    logger.info("Starting pretraining...")
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10  # Stop if no improvement for 10 epochs

    for epoch in range(1, config.epochs + 1):
        # train from one epoch 
        train_loss = train_epoch(model, data_obj["train_dataloader"], optimizer, epoch, logger, config)

        # Validate
        val_loss = validate(model, data_obj["val_dataloader"], logger, config)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save best model
            model_save_path = os.path.join(exp_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_save_path)
            logger.info(f"Best model saved at epoch {epoch} with validation loss {val_loss:.6f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Save final model after all epochs
    final_model_path = os.path.join(exp_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    
    logger.info("Pretraining completed.")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Models saved in {exp_dir}")
  
       

    



    
    
    ### modeling ####