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
from data_loader import load_classification_data
# from model.classification import TSClassification
import sys

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
    data_obj = load_classification_data(config)
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    test_loader = data_obj["test_dataloader"]

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        data, time_steps, mask, labels = batch['data'], batch['time_steps'], batch['mask'], batch['labels']
        print('checking data shape', data.shape,labels.shape)
        print(labels)
        sys.exit()
    
    
    #### modeling ####