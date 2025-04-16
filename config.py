import os
import json
import torch

class Config:
    def __init__(self):
        # Basic settings
        self.seed = 42
        self.experiment_name = "mtan_subset_train_from_scratch"
        self.description = "mask in both raw and latent space, and discuss the finetune architecture"
        self.log_dir = "/home/yw573/rds/hpc-work/nips25/logs"
        
        # Training parameters
        self.epochs = 30
        self.batch_size = 32  
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hid_dim = 128  # Hidden dimension
        self.te_dim = 16    # Time embedding dimension or 16?
        self.mask_ratio = 0.5  # Ratio of patches to mask
        self.patch_size = 16   # Number of patches
        self.num_heads = 4  # Number of attention heads
        self.embed_time = 16  # Embedding dimension
        
        # Dataset parameters
        self.dataset_name = "mimic"  # physionet,activity
        self.quantization = 0.0  # Quantization for the dataset
        self.n = 10000000  # Number of samples
        self.percentage_tp_to_sample = 0.5  # Percentage of time points to sample
        self.patience = 5  # Early stopping patienc
        self.interp_lr = 0.0001

        ## random_mask
        self.max_len = 512
        self.mask_ratio_per_seg = 0.15
        self.segment_num = 3

        # Downstream 
        self.pretrained_path = "/home/yw573/rds/hpc-work/nips25/logs/mtan_complexmask_20250409_171706/best_model.pth"
        self.masking_mode = "end"  # Options: "end", "beginning", "middle", "random_blocks", "random"
        self.num_blocks = 2  # Only used for "random_blocks" mode
        self.history = 24 ## for physionet/mimic, 48 hrs, for physionet 3000
        self.pred_window = 1000
        

        
    def load(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if key == 'device':
                self.device = torch.device(value)
            else:
                setattr(self, key, value)
                
    def save(self, config_path):
        """Save configuration to JSON file"""
        config_dict = self.__dict__.copy()
        
        # Convert non-serializable objects
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)