import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from torch import nn

'''
This version contains both raw signal - masking
and latent space - masking.
The model is trained to reconstruct the original signal
and the latent representation.
'''

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), # to embed the query
                                      nn.Linear(embed_time, embed_time), # to embed the key
                                      nn.Linear(input_dim*num_heads, nhidden)]) # to embed attention weighted values
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.to(query.device).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn.to(query.device)*value.unsqueeze(-3).to(query.device), -2), p_attn.to(query.device)
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)
    

class ChannelIndependentTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(ChannelIndependentTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim  # This is 2 for value+mask per channel
        self.nhidden = nhidden
        
        # Same query/key embeddings
        self.linears = nn.ModuleList([
            nn.Linear(embed_time, embed_time),          # to embed the query
            nn.Linear(embed_time, embed_time),          # to embed the key
            nn.Linear(input_dim*num_heads, nhidden)])   # to embed attention weighted values
        
    # The attention method stays the same
    def attention(self, query, key, value, mask=None, dropout=None):
        # Same implementation as original
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.to(query.device).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn.to(query.device)*value.unsqueeze(-3).to(query.device), -2), p_attn.to(query.device)
    
    # Forward method stays the same
    def forward(self, query, key, value, mask=None, dropout=None):
        # Same implementation as original
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                    for l, x in zip(self.linears, (query, key))]

        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        print('check x shape:', x.shape)
        sys.exit()
        return self.linears[-1](x)
    

class gMAE(nn.Module):
    def __init__(self, config):
        super(gMAE, self).__init__()
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.hid_dim = config.hid_dim
        self.te_dim = config.te_dim
        self.embed_time = config.embed_time

        # Time embedding layers
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        

        # Additional parameters
        self.latent_mask_ratio = config.latent_mask_ratio if hasattr(config, 'mask_ratio') else 0.5
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

        # Add a learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim // self.patch_size))
        # Initialize mask token with small random values
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional embedding for patches
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.patch_size, self.hid_dim // self.patch_size))
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        
        ### initialization of mTAN
        self.time_att = multiTimeAttention(2, self.hid_dim, self.embed_time, config.num_heads)
        self.cls_query = torch.linspace(0, 1., 128)
        self.periodic = nn.Linear(1, self.te_dim-1)
        self.linear = nn.Linear(1, 1)

        # Replace mean pooling with attention pooling
        self.query_attention = nn.Parameter(torch.randn(1, 1, self.hid_dim))

        # Create a signal projector that adapts to sequence length
        self.max_seq_len = config.max_len
        self.signal_projector = nn.Linear(self.hid_dim, self.max_seq_len)

        self.mask_ori = config.mask_ori
        

    def LearnableTE(self, tt):
        """
        Learnable time embedding.
        :param tt: (B * D, L, 1) tensor of time steps.
        :return: (B * D, L, te_dim) tensor of time embeddings.
        """
        out1 = self.te_scale(tt)  # Linear time embedding
        out2 = torch.sin(self.te_periodic(tt))  # Periodic time embedding
        return torch.cat([out1, out2], -1)
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
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
    
    def patchify_and_mask(self, latent_representation):
        """Patchify latent representation and create mask."""
        B, D, _ = latent_representation.shape
        
        # Patchify the latent representation
        patched = latent_representation.view(B, D, self.patch_size, self.patch_dim)
        orig_patches = patched.clone()  # Save for loss calculation
        
        # Reshape to (B*D, patch_size, patch_dim) for processing
        patched_flat = patched.reshape(-1, self.patch_size, self.patch_dim)
        
        # Add positional embeddings (essential for proper MAE functioning)
        patched_flat = patched_flat + self.patch_pos_embed
        
        # Generate random mask for each sample-channel
        num_samples = B * D
        num_patches = self.patch_size
        mask_length = int(num_patches * self.latent_mask_ratio)
        
        # Create mask indices
        rand_indices = torch.rand((num_samples, num_patches), device=self.device).argsort(dim=1)
        mask_indices = rand_indices[:, :mask_length]
        
        # Create Boolean mask (True = keep, False = mask)
        keep_mask = torch.ones((num_samples, num_patches), dtype=torch.bool, device=self.device)
        batch_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, mask_length)
        keep_mask[batch_indices, mask_indices] = False
        
        # Store information for later processing
        return {
            'patched_flat': patched_flat,
            'orig_patches': orig_patches,
            'keep_mask': keep_mask,
            'mask_indices': mask_indices,
            'batch_indices': batch_indices,
            'num_samples': num_samples,
            'num_patches': num_patches
        }

    def reconstruct_full_sequence(self, patch_info, encoded_visible, visible_mask):
        """Reconstruct full sequence including masked tokens."""
        num_samples = patch_info['num_samples']
        num_patches = patch_info['num_patches']
        keep_mask = patch_info['keep_mask']
        
        # Create full tensor for reconstruction
        full_seq = torch.zeros(num_samples * num_patches, self.patch_dim, device=self.device)
        
        # 1. Place encoded visible patches
        visible_indices = torch.nonzero(visible_mask).squeeze(-1)
        decoded_visible = self.decoder(encoded_visible)
        full_seq[visible_indices] = decoded_visible
        
        # 2. Generate masked tokens with positional information
        mask_indices_flat = torch.nonzero(~keep_mask.view(-1)).squeeze(-1)
        sample_indices = torch.div(mask_indices_flat, num_patches, rounding_mode='floor')
        patch_positions = mask_indices_flat % num_patches
        
        # Create mask tokens with positional information
        mask_tokens = self.mask_token.expand(len(mask_indices_flat), -1)
        
        # Add position embedding to mask tokens (extract relevant positions)
        pos_embed = self.patch_pos_embed[0, patch_positions]
        mask_tokens = mask_tokens + pos_embed
        
        # Decode mask tokens
        decoded_masks = self.decoder(mask_tokens)
        
        # Place in full sequence
        full_seq[mask_indices_flat] = decoded_masks
        
        # Reshape back to original format
        return full_seq.view(num_samples, num_patches, self.patch_dim)

    def forward(self, value, truth_time_steps, mask):
        """
        Forward pass for gMAE with faithful MAE implementation.
        """
        B, L, D = value.shape
        original_value = value.clone()
        original_mask = mask.clone()

        if self.mask_ori:
            mask = original_mask[:, :, :D]  # Only keep the first D channels for masking
            interp_mask = original_mask[:, :, D:]  # Keep the remaining channels for interpolation

        # Create time embeddings
        key = self.learn_time_embedding(truth_time_steps).to(self.device)  # (B, L, te_dim)
        query = self.learn_time_embedding(self.cls_query.unsqueeze(0)).to(self.device)  # (1, 128, te_dim)
                
        # Process all D channels at once by reshaping 
        value_reshaped = value.permute(0, 2, 1).reshape(B*D, L, 1) # (B, L, D) to (B*D, L, 1)
        mask_reshaped = mask.permute(0, 2, 1).reshape(B*D, L, 1) # (B, L, D) to (B*D, L, 1)
        
        # Concatenate value and mask for each channel, follow mTAN
        x_all = torch.cat((value_reshaped, mask_reshaped), 2)  # (B*D, L, 2)
        mask_all = torch.cat((mask_reshaped, mask_reshaped), 2)  # (B*D, L, 2)
        key_expanded = key.unsqueeze(1).repeat(1, D, 1, 1).reshape(B*D, L, -1) # Repeat key and query for each channel
        query_expanded = query.repeat(B*D, 1, 1)
        
        # Process all channels at once through time attention
        output = self.time_att(query_expanded, key_expanded, x_all, mask_all)  # (B*D, 128, hid_dim)
        
        # Reshape output back to (B, D, hid_dim)
        output = output.reshape(B, D, -1)
        
        
        # ############## PATCHIFYING LATENT REPRESENTATION #############

        patch_info = self.patchify_and_mask(output)

        visible_mask = patch_info['keep_mask'].view(-1)  # Flatten to (B*D*patch_size)
        visible_patches = patch_info['patched_flat'].view(-1, self.patch_dim)[visible_mask]
        
        # Process through transformer encoder
        encoded_visible = self.transformer_encoder(visible_patches)

        full_reconstruction_flat = self.reconstruct_full_sequence(
            patch_info, encoded_visible, visible_mask
        )
        full_reconstruction = full_reconstruction_flat.view(B, D, self.patch_size, self.patch_dim)
 
               # Step 5: Project back to signal space
        reconstructed_latent = full_reconstruction.reshape(B * D, self.hid_dim)
        reconstructed_signal_flat = self.signal_projector(reconstructed_latent)
        reconstructed_signals = reconstructed_signal_flat.view(B, D, L).transpose(1, 2)  # (B, L, D)
        
        # Compute latent space loss on masked patches
        keep_mask = patch_info['keep_mask']
        orig_patches = patch_info['orig_patches']
        masked_patches_orig = orig_patches[~keep_mask.view(B, D, self.patch_size)]
        masked_patches_recon = full_reconstruction[~keep_mask.view(B, D, self.patch_size)]
        latent_loss = F.mse_loss(masked_patches_recon, masked_patches_orig)
        
        # Compute signal space loss
        if self.mask_ori:
            signal_loss = torch.sum(((reconstructed_signals - original_value) * interp_mask) ** 2) / interp_mask.sum()
        else:
            signal_loss = F.mse_loss(reconstructed_signals, original_value)
        
        # Combined loss with proper clamping
        alpha = 0.7  # Weight for signal vs latent loss
        signal_loss = torch.clamp(signal_loss, min=0.0, max=1e6) 
        latent_loss = torch.clamp(latent_loss, min=0.0, max=1e6)
        combined_loss = alpha * signal_loss + (1-alpha) * latent_loss

        
        return {
            'loss': combined_loss,
            'reconstructed_signal': reconstructed_signals,
            'encoded_representation': output,
            'latent_loss': latent_loss,
            'signal_loss': signal_loss
        }