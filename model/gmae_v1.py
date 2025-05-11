import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from torch import nn

class gMAE(nn.Module):
    def __init__(self, config):
        super(gMAE, self).__init__()
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.hid_dim = config.hid_dim
        self.te_dim = config.te_dim

        # Time embedding layers
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        # TTCN layers for independent channel mapping
        input_dim = 1 + self.te_dim
        self.ttcn_dim = self.hid_dim
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, input_dim * self.ttcn_dim, bias=True)
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

        # Additional parameters
        self.mask_ratio = config.mask_ratio if hasattr(config, 'mask_ratio') else 0.5
        self.patch_size = config.patch_size

        # Transformer encoder for processing visible patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim // self.patch_size,
            nhead=8,
            dim_feedforward=4 * (self.hid_dim // self.patch_size),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim // self.patch_size, 4 * (self.hid_dim // self.patch_size)),
            nn.GELU(),
            nn.Linear(4 * (self.hid_dim // self.patch_size), self.hid_dim // self.patch_size)
        )

        # Add a signal projection layer that maps from latent space to original signal
        # self.signal_projector = nn.Linear(self.hid_dim, config.L)

        # Create the signal projector once during initialization with a default size
        # We'll handle variable length sequences in forward pass
        self.base_signal_projector = nn.Linear(self.hid_dim, 128)  # Base projector

    def LearnableTE(self, tt):
        """
        Learnable time embedding.
        :param tt: (B * D, L, 1) tensor of time steps.
        :return: (B * D, L, te_dim) tensor of time embeddings.
        """
        out1 = self.te_scale(tt)  # Linear time embedding
        out2 = torch.sin(self.te_periodic(tt))  # Periodic time embedding
        return torch.cat([out1, out2], -1)

    def TTCN(self, X, mask):
        """
        Temporal-Temporal Convolutional Network (TTCN) for independent channel mapping.
        :param X: (B * D, L, F) input tensor.
        :param mask: (B * D, L, 1) mask tensor.
        :return: (B, D, hid_dim) tensor after mapping each channel independently.
        """
        B_D, L, F_in = X.shape  # B_D = B * D
        Filter = self.Filter_Generators(X)  # (B * D, L, F_in * ttcn_dim)
        Filter_mask = Filter * mask + (1 - mask) * (-1e8)  # Apply mask
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # Normalize along the sequence dimension
        Filter_seqnorm = Filter_seqnorm.view(B_D, L, self.ttcn_dim, -1)  # (B * D, L, ttcn_dim, F_in)
        X_broad = X.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)  # (B * D, L, ttcn_dim, F_in)
        ttcn_out = torch.sum(X_broad * Filter_seqnorm, dim=(-3, -1))  # Combine sums
        h_t = torch.relu(ttcn_out + self.T_bias)  # (B * D, ttcn_dim)

        return h_t

    def forward(self, value, truth_time_steps, mask):
        """
        Forward pass for gMAE with vectorized channel processing.
        """
        B, L, D = value.shape
        original_value = value.clone()

        # Process irregular data with TTCN to get latent representation
        # here we treat each channel as an independent sample
        X = value.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)
        truth_time_steps = truth_time_steps.unsqueeze(-1).expand(B, L, D)
        truth_time_steps = truth_time_steps.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)
        mask_reshaped = mask.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)

        # Positional embedding
        te_his = self.LearnableTE(truth_time_steps)  # (B * D, L, te_dim)
        X = torch.cat([X, te_his], dim=-1)  # (B * D, L, F)

        # Apply TTCN
        output = self.TTCN(X, mask_reshaped)  # (B*D, hid_dim)
        output = output.view(B, D, self.ttcn_dim)  # (B, D, hid_dim)

        # Patchify the latent representation
        patched_output = output.view(B, D, self.patch_size, self.hid_dim // self.patch_size)
        orig_patches = patched_output.clone()  # Save for loss calculation

        # Reshape to combine batch and channel dimensions for vectorized processing
        # From (B, D, patch_size, hid_dim//patch_size) to (B*D, patch_size, hid_dim//patch_size)
        patched_output_flat = patched_output.reshape(-1, self.patch_size, self.hid_dim // self.patch_size)
        
        # Create masking tensors for all samples and channels at once
        num_samples = B * D
        num_patches = self.patch_size
        mask_length = int(num_patches * self.mask_ratio)
        
        # Create random masks for all samples at once
        rand_perm = torch.rand((num_samples, num_patches), device=self.device).argsort(dim=1)
        mask_indices = rand_perm[:, :mask_length]
        # Create boolean mask
        bool_mask_flat = torch.ones(num_samples, num_patches, dtype=torch.bool, device=self.device)
        
        # Use advanced indexing to set masked positions
        batch_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, mask_length)
        bool_mask_flat[batch_indices, mask_indices] = False
        
        # Create a mask for selecting visible patches from the flattened tensor
        # This creates a binary mask of shape (B*D*patch_size) where 1's are for visible patches
        select_visible = bool_mask_flat.view(-1)
        
        # Use the mask to select all visible patches at once
        visible_patches = patched_output_flat.view(-1, self.hid_dim // self.patch_size)[select_visible]
        
        # Process all visible patches through transformer at once
        encoded_visible = self.transformer_encoder(visible_patches)
        decoded_visible = self.decoder(encoded_visible)
        
        # Create full reconstruction tensor
        full_reconstruction_flat = torch.zeros_like(patched_output_flat)
        
        # Place decoded patches back into the reconstruction tensor
        # First we create empty tensors to hold the results for visible and masked positions
        visible_indices = torch.nonzero(select_visible).squeeze(-1)
        full_reconstruction_flat.view(-1, self.hid_dim // self.patch_size)[visible_indices] = decoded_visible
        
        # For masked patches, compute average encoding per sample-channel
        # Create a tensor to store the average encoded representation for each sample-channel
        avg_encodings = torch.zeros(num_samples, self.hid_dim // self.patch_size, device=self.device)
        
        # For each sample-channel, compute the mean of its visible patches
        visible_count = torch.sum(bool_mask_flat, dim=1)  # Count of visible patches per sample
        
        # Use index_add to efficiently compute sum of encodings for each sample-channel
        encoded_visible_with_index = torch.zeros(num_samples, self.hid_dim // self.patch_size, device=self.device)
        sample_indices = torch.nonzero(select_visible).squeeze(-1) // num_patches
        encoded_visible_with_index.index_add_(0, sample_indices, encoded_visible)
        
        # Replace the loops with vectorized operations
        non_zero_counts = torch.clamp(visible_count, min=1).unsqueeze(1)  # Avoid division by zero
        avg_encodings = encoded_visible_with_index / non_zero_counts
        
        # Decode average encodings
        avg_decoded = self.decoder(avg_encodings)
        
        # Vectorize filling masked positions
        masked_indices = torch.nonzero(~bool_mask_flat.view(-1)).squeeze(-1)
        sample_indices_for_masked = torch.div(masked_indices, num_patches, rounding_mode='floor')
        decoded_to_use = avg_decoded[sample_indices_for_masked]
        full_reconstruction_flat.view(-1, self.hid_dim // self.patch_size)[masked_indices] = decoded_to_use
        
        # Reshape back to (B, D, patch_size, hid_dim//patch_size)
        full_reconstruction = full_reconstruction_flat.view(B, D, self.patch_size, self.hid_dim // self.patch_size)
        
        # Create signal projector
        signal_projector = nn.Linear(self.hid_dim, L).to(self.device)
        
        # Reshape and project all channels at once
        reconstructed_latent = full_reconstruction.reshape(B * D, self.hid_dim)
        reconstructed_signal_flat = signal_projector(reconstructed_latent)
        reconstructed_signals = reconstructed_signal_flat.view(B, D, L).transpose(1, 2)  # (B, L, D)
        
        # Compute latent space loss on masked patches
        masked_patches_orig = orig_patches[~bool_mask_flat.view(B, D, self.patch_size)]
        masked_patches_recon = full_reconstruction[~bool_mask_flat.view(B, D, self.patch_size)]
        latent_loss = F.mse_loss(masked_patches_recon, masked_patches_orig)
        
        # Compute signal space loss
        signal_loss = F.mse_loss(reconstructed_signals, original_value)
        
        # Combined loss
        alpha = 0.7  # Weight for signal vs latent loss
        combined_loss = alpha * signal_loss + (1-alpha) * latent_loss

        
        
        return {
            'loss': combined_loss,
            'reconstructed_signal': reconstructed_signals,
            'encoded_representation': output
        }