from data_processing.physionet_full import PhysioNet_full, get_data_min_max_inter
from data_processing.physionet_classification import PhysioNet_classify, get_data_min_max
from data_processing.person_activity import PersonActivity, Activity_time_chunk
from data_processing.mimic import MIMIC
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys

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


# def normalize_masked_data(data, mask, att_min, att_max):
#     """
#     Normalize data using min-max scaling with masking.
#     """
#     data = (data - att_min) / (att_max - att_min + 1e-6)
#     data = data * mask
#     return data

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

def normalize_masked_tp(data, att_min, att_max):
	scale = att_max - att_min
	scale = scale + (scale == 0) * 1e-8
	# we don't want to divide by zero
	if (scale != 0.).all():
		data_norm = (data - att_min) / scale
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm

def variable_time_collate_fn_fore(batch, config, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 

	for b, (record_id, tt, vals, mask) in enumerate(batch):
		n_observed_tp = torch.lt(tt, config.history).sum()
		observed_tp.append(tt[:n_observed_tp])
		observed_data.append(vals[:n_observed_tp])
		observed_mask.append(mask[:n_observed_tp])
		
		predicted_tp.append(tt[n_observed_tp:])
		predicted_data.append(vals[n_observed_tp:])
		predicted_mask.append(mask[n_observed_tp:])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)


	observed_data = normalize_masked_data(observed_data, observed_mask, 
			att_min = data_min, att_max = data_max)
	predicted_data = normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)
	
	observed_tp = normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	
	return data_dict


def load_forecasting_data(config):
    """
    Load the data based on the configuration provided.
    """
    device = config.device
    #/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/physionet_label
    if config.dataset_name in ["physionet", "mimic"]:
        if config.dataset_name == "physionet":
            total_dataset = PhysioNet_full(
                '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full', ## modify the path here
                quantization=config.quantization,
                download=True,
                n_samples=config.n,
                device=device
            )
        elif config.dataset_name == "mimic":
             total_dataset = MIMIC('/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_old', n_samples = config.n, device = device)

        # only use subset of 1000 samples
        
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

    

        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max_inter(train_data, device)

        # Create data loaders to sperate data based on their time length
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="train",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, device, classify=False,data_type="val",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, device, classify=False,data_type="test",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        
        data_obj = {
					"train_dataloader": train_dataloader, 
					"val_dataloader": val_dataloader,
					"test_dataloader": test_dataloader,
					"input_dim": train_data[0][2].shape[1],
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional
    elif config.dataset_name == "activity":
        config.pred_window = 1000 # predict future 1000 ms
        # Load PhysioNet dataset
        total_dataset = PersonActivity(
            '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/activity_classification/PersonActivity/PersonActivity_tpach',
            download=True,
            n_samples=config.n,
            device=device
        )

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))


        # Get data min, max, and time max for normalization
        data_min, data_max, _= get_data_min_max_inter(total_dataset, device)
        time_max = torch.tensor(config.history + config.pred_window)
        print('manual set time_max:', time_max)

        train_data = Activity_time_chunk(train_data, config, device)
        val_data = Activity_time_chunk(val_data, config, device)
        test_data = Activity_time_chunk(test_data, config, device)
        print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
        
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="train",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, device, classify=False, data_type="val",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, device, classify=False, data_type="test",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        
        data_obj = {
					"train_dataloader": train_dataloader, 
					"val_dataloader": val_dataloader,
					"test_dataloader": test_dataloader,
					"input_dim": train_data[0][2].shape[1],
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional


    return data_obj
    

def load_interpolation_data(config):
    """
    Load the data based on the configuration provided.
    """
    device = config.device
    #/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/physionet_label
    if config.dataset_name == "physionet":
        # Load PhysioNet dataset
        total_dataset = PhysioNet_full(
            '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full',
            quantization=config.quantization,
            download=True,
            n_samples=config.n,
            device=device
        )

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))


        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max_inter(total_dataset, device)

        # Create data loaders
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

    elif config.dataset_name == "activity":
        # Load PhysioNet dataset
        total_dataset = PersonActivity(
            '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/activity_classification/PersonActivity',
            download=True,
            n_samples=config.n,
            device=device
        )

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))


        # Get data min, max, and time max for normalization
        data_min, data_max, _= get_data_min_max(total_dataset, device)

        # Create data loaders
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max
        }
    return data_obj



def load_classification_data(config):
    """
    Load data specifically for PhysioNet mortality prediction classification task
    """
    device = config.device
    
    if config.dataset_name in ["physionet", "mimic"]:
        if config.dataset_name == "physionet":
        # Load PhysioNet dataset
            total_dataset = PhysioNet_classify(
                '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_classification', ##### change here to your path
                quantization=config.quantization,
                download=True,  # Data already exists
                n_samples=config.n,
                device=device
            )
        elif config.dataset_name == "mimic":
            total_dataset = MIMIC('/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_classification', n_samples = config.n, device = device)
        
        # Use subset if specified (for faster testing)
        # if hasattr(config, 'use_subset') and config.use_subset:
        #     subset_size = min(1000, len(total_dataset))
        #     total_dataset = total_dataset[:subset_size]
        
        print("Dataset size:", len(total_dataset))
        
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max(total_dataset, device)
        
        # Create data loaders with classification-specific collate function
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
             
        }
        
    elif config.dataset_name == "activity":
        total_dataset = PersonActivity('/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/activity_classification',
                                 download=True, n_samples=config.n, device=device)
          
          # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max(total_dataset, device)
        
        # Create data loaders with classification-specific collate function
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
             
        }
        
    return data_obj

def variable_time_collate_classification(batch, device=torch.device("cpu"), data_min=None, data_max=None,activity = False):
    """
    Collate function for classification tasks with PhysioNet mortality labels
    """
    # Remove the debugging code that's causing sys.exit()
    D = batch[0][2].shape[1]  # Number of features
    N = batch[0][-1].shape[1] if activity else 1
    # Get max sequence length in the batch
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = max(len_tt)
    
    # Initialize tensors with zeros
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if activity:
        combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)], dtype=torch.long).to(device)
    
    # Fill tensors with data
    # can we still save the record_id to the batch

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
 
        # Handle mortality label (binary classification)
        if labels is not None:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:   
                # combined_labels[b] = labels.long().to(device)  # Convert to long for CrossEntropyLoss
                combined_labels[b] = torch.tensor(labels, dtype=torch.long).to(device)
        
    
    # Normalize data
    if not activity:
        enc_combined_vals = normalize_masked_data(enc_combined_vals, enc_combined_mask, data_min, data_max)
    
    # Normalize time steps
    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
    
    return {
        "data": enc_combined_vals,
        "time_steps": enc_combined_tt,
        "mask": enc_combined_mask,
        "labels": combined_labels,
        # "record_id": record_id

    }