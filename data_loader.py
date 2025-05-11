# import signal

# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException("Operation timed out")

# Set a timeout for potentially blocking operations
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(30)  # 30 second timeout

# print("Starting module import...")

# try:
#     from data_processing.physionet_full import PhysioNet_full, get_data_min_max_inter
#     print("Imported PhysioNet_full")
#     # Reset the alarm
#     signal.alarm(0)
# except TimeoutException:
#     print("Import timed out - likely hanging on file operations")
# except Exception as e:
#     print(f"PhysioNet_full import error: {e}")
    
# try:
#     # from data_processing.physionet_classification import PhysioNet_classify, get_data_min_max ## also will stuck in the gpu setting..
#     # from data_processing.person_activity import PersonActivity, Activity_time_chunk
#     # from data_processing.mimic import MIMIC
#     print("Imported MIMIC")
# except Exception as e:
#     print(f"MIMIC import error: {e}")

# print("All imports completed")

from data_processing.utils import get_data_min_max_inter
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys
import time

def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]  # Number of features
    N = batch[0][-1].shape[0] if activity else 1  # Number of labels
    ## because for each_id, they show different length of time series, then here we want to make sure 
    # all batches have the same length of time series, then we padding to ensure the shorter sequences are extended
    # to match the longest sequences in the batch
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = max(len_tt)

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if not activity:
        for b, (record_id, tt, vals, mask) in enumerate(batch):
            currlen = tt.size(0)
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)
    else:
        for b, (record_id, tt, vals, mask, _) in enumerate(batch):
            currlen = tt.size(0)
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)


    # Normalize data
    if not activity:
        enc_combined_vals = normalize_masked_data(enc_combined_vals, enc_combined_mask, data_min, data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "data": enc_combined_vals,  # (batch_size, maxlen, D)
        "time_steps": enc_combined_tt,  # (batch_size, maxlen)
        "mask": enc_combined_mask,  # (batch_size, maxlen, D)
    }


    return data_dict



def normalize_masked_data(data, mask, att_min, att_max):
	scale = att_max - att_min
	scale = scale + (scale == 0) * 1e-8
	# we don't want to divide by zero
	if (scale != 0.).all(): 
		data_norm = (data - att_min) / scale
	else:
		raise Exception("Zero!")

	# set masked out elements back to zero 
	data_norm[mask == 0] = 0

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm


class MultiDatasetIterator:
    """Iterator that alternates between multiple datasets"""
    def __init__(self, dataloaders, cycle_mode=False):
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.dataset_idx = 0  # Track which dataset we're using
        self.exhausted_iterators = set()  # Track which iterators are exhausted
        self.cycle_mode = cycle_mode  # Whether to cycle indefinitely
        
    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.dataset_idx = 0
        self.exhausted_iterators = set()
        return self
        
    def __next__(self):
        start_time = time.time()
        
        # Rest of your existing __next__ method...
        try:
            # Timeout check to prevent infinite loops
            if time.time() - start_time > 10:  # 10 second timeout
                print(f"Warning: Dataset iteration timeout for dataset {self.dataset_idx}")
                self.dataset_idx = (self.dataset_idx + 1) % len(self.dataloaders)
                return self.__next__()
                
            # Your existing code
            batch = next(self.iterators[self.dataset_idx])
            batch["dataset_idx"] = self.dataset_idx
            return batch
        except StopIteration:
            # Mark this iterator as exhausted
            self.exhausted_iterators.add(self.dataset_idx)
            
            if self.cycle_mode:
                # Reset this iterator
                self.iterators[self.dataset_idx] = iter(self.dataloaders[self.dataset_idx])
                
            # Move to next dataset
            self.dataset_idx = (self.dataset_idx + 1) % len(self.dataloaders)
            
            # If we've checked all datasets and they're all exhausted, stop
            if len(self.exhausted_iterators) == len(self.dataloaders):
                if not self.cycle_mode:
                    raise StopIteration
                else:
                    # Reset for next cycle
                    self.exhausted_iterators = set()
                
            # Try the next dataset
            return self.__next__()

    def __len__(self):
        """Return combined length of all dataloaders"""
        return sum(len(dl) for dl in self.dataloaders)

def load_data(config):
    """Load each dataset separately but combine them for training"""
    device = torch.device('cpu')
    datasets_info = []
    train_dataloaders = []
    val_dataloaders = []
    
    # Process PhysioNet dataset
    # physio_dataset = PhysioNet_full(
    #     '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full',
    #     quantization=config.quantization,
    #     download=False,
    #     n_samples=config.n,
    #     device=torch.device('cpu')
    # )

    data_a = torch.load('/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full/processed/set-a_0.0.pt', map_location='cpu')
    data_b = torch.load('/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full/processed/set-b_0.0.pt', map_location='cpu')
    data_c = torch.load('/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full/processed/set-c_0.0.pt', map_location='cpu')

    physio_dataset = data_a + data_b + data_c

    physio_dataset = physio_dataset[:1000]  # Use a subset for faster testing
    
    # Split PhysioNet
    physio_train, physio_val = model_selection.train_test_split(
        physio_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    
    # Get normalization parameters for PhysioNet
    physio_min, physio_max, physio_time_max = get_data_min_max_inter(physio_dataset, device)
    
    # Create PhysioNet dataloaders
    physio_train_loader = DataLoader(
        physio_train, batch_size=config.batch_size, shuffle=True,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=physio_min, data_max=physio_max)
    )
    physio_val_loader = DataLoader(
        physio_val, batch_size=config.batch_size, shuffle=False,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=physio_min, data_max=physio_max)
    )
    
    train_dataloaders.append(physio_train_loader)
    val_dataloaders.append(physio_val_loader)
    
    datasets_info.append({
        "name": "physionet",
        "input_dim": physio_train[0][2].shape[1],
        # "data_min": physio_min,
        # "data_max": physio_max,
        # "time_max": physio_time_max
    })
    
    # Process MIMIC dataset similarly...
    # mimic_dataset = MIMIC(
    #     '/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_old',
    #     n_samples=config.n, device=torch.device('cpu')
    # )

    mimic_dataset = torch.load('/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_old/processed/mimic.pt', map_location='cpu')

    mimic_dataset = mimic_dataset[:2000]
    
    # Split MIMIC
    mimic_train, mimic_val = model_selection.train_test_split(
        mimic_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    
    # Get normalization parameters for MIMIC
    mimic_min, mimic_max, mimic_time_max = get_data_min_max_inter(mimic_dataset, device)
    
    # Create MIMIC dataloaders
    mimic_train_loader = DataLoader(
        mimic_train, batch_size=config.batch_size, shuffle=True,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=mimic_min, data_max=mimic_max)
    )
    mimic_val_loader = DataLoader(
        mimic_val, batch_size=config.batch_size, shuffle=False,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=mimic_min, data_max=mimic_max)
    )
    
    train_dataloaders.append(mimic_train_loader)
    val_dataloaders.append(mimic_val_loader)
    
    datasets_info.append({
        "name": "mimic",
        "input_dim": mimic_train[0][2].shape[1],
        # "data_min": mimic_min,
        # "data_max": mimic_max,
        # "time_max": mimic_time_max
    })
    
    # Create combined iterators
    multi_train_iterator = MultiDatasetIterator(train_dataloaders)
    multi_val_iterator = MultiDatasetIterator(val_dataloaders)
    
    return {
        "train_dataloader": multi_train_iterator,
        "val_dataloader": multi_val_iterator,
        "datasets_info": datasets_info
    }