import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from torch import nn

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
        return self.linears[-1](x)
    

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

        # Add a learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim // self.patch_size))
        # Initialize mask token with small random values
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional embedding for patches
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.patch_size, self.hid_dim // self.patch_size))
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        self.periodic = nn.Linear(1, self.te_dim-1)
        self.linear = nn.Linear(1, 1)

        # self.time_att = multiTimeAttention(2, self.hid_dim, self.te_dim, config.num_heads)
        self.time_att = ChannelIndependentTimeAttention(1, self.hid_dim, self.te_dim, config.num_heads)
        self.cls_query = torch.linspace(0, 1., 128)

        # Replace mean pooling with attention pooling
        self.query_attention = nn.Parameter(torch.randn(1, 1, self.hid_dim))
        

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

    def forward(self, value, truth_time_steps, mask):
        """
        Forward pass for gMAE with faithful MAE implementation.
        truth_time_steps: (B, L) tensor of time steps.
        """
        B, L, D = value.shape
        original_value = value.clone()

        # Process irregular data with TTCN to get latent representation
        # here we treat each channel as an independent sample
        X = value.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)
        time_steps = truth_time_steps.clone()
        query_time_steps = time_steps
        truth_time_steps = truth_time_steps.unsqueeze(-1).expand(B, L, D)
        truth_time_steps = truth_time_steps.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)
        mask_reshaped = mask.permute(0, 2, 1).reshape(-1, L, 1)  # (B * D, L, 1)

        # Positional embedding
        te_his = self.LearnableTE(truth_time_steps)  # (B * D, L, te_dim)
        X = torch.cat([X, te_his], dim=-1)  # (B * D, L, F)

        # Apply TTCN
        output = self.TTCN(X, mask_reshaped)  # (B * D, ttcn_dim)
        output = output.view(B, D, self.ttcn_dim)  # (B, D, ttcn_dim)

        # Apply mtan - FIX THE IMPLEMENTATION HERE
        # time_steps_expanded = time_steps.unsqueeze(1).expand(-1, D, -1)  # (B, D, L)
        # time_steps_reshaped = time_steps_expanded.reshape(-1, L)          # (B*D, L)
        
        # # # Create time embeddings
        # key = self.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        # time_query = self.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        # cls_query = self.learn_time_embedding(self.cls_query.unsqueeze(0).to(self.device))  # (1, 128, te_dim)
        
        # # IMPORTANT - Correctly prepare inputs for mTAN
        # # Only include the value itself, not time embeddings (those go in key/query)
        # X_value = X[:, :, :1]  # Just the actual values (B*D, L, 1)
        
        # # IMPORTANT - Use mask directly, don't concatenate it twice
        # #  The mask should match the value dimensions
        # mtan_out = self.time_att(
        #     cls_query.repeat(B*D, 1, 1),  # Repeat cls_query for each channel
        #     key,                          # Time embeddings as key
        #     X_value,                      # Values only
        #     mask_reshaped                 # Use mask directly
        # )
        # # simple average: 
        # # mtan_out = mtan_out.view(B, D, 128, -1)
        # # output = torch.mean(mtan_out, dim=2)  # Average over the 128 query timestamps
   
        # # mtan_out shape should be (B*D, 128, hidden_size)
        # # Replace mean pooling with attention pooling
        # attention_weights = F.softmax(torch.bmm(
        #     self.query_attention.repeat(B*D, 1, 1), 
        #     mtan_out.transpose(1, 2)
        # ), dim=-1)
        # output = torch.bmm(attention_weights, mtan_out).squeeze(1).view(B, D, -1)

        ############## PATCHIFYING LATENT REPRESENTATION #############

        # Patchify the latent representation
        patched_output = output.view(B, D, self.patch_size, self.hid_dim // self.patch_size)
        orig_patches = patched_output.clone()  # Save for loss calculation

        # Reshape to combine batch and channel dimensions for vectorized processing
        patched_output_flat = patched_output.reshape(-1, self.patch_size, self.hid_dim // self.patch_size)
        
        # Create masking tensors for all samples and channels at once
        num_samples = B * D
        num_patches = self.patch_size
        mask_length = int(num_patches * self.mask_ratio)
        
        # Create random masks for all samples at once
        rand_perm = torch.rand((num_samples, num_patches), device=self.device).argsort(dim=1)
        mask_indices = rand_perm[:, :mask_length]
        
        # Create boolean mask (True for visible, False for masked)
        bool_mask_flat = torch.ones(num_samples, num_patches, dtype=torch.bool, device=self.device)
        batch_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, mask_length)
        bool_mask_flat[batch_indices, mask_indices] = False
        
        # Get visible patch indices and sample indices for each visible patch
        visible_indices = torch.nonzero(bool_mask_flat)  # (num_visible, 2) with [sample_idx, patch_idx]
        sample_indices_visible = visible_indices[:, 0]
        patch_indices_visible = visible_indices[:, 1]
        
        # Extract visible patches
        visible_patches = patched_output_flat[sample_indices_visible, patch_indices_visible]
        
        # Add positional embeddings to visible patches (based on their original positions)
        pos_embed_visible = self.patch_pos_embed[0, patch_indices_visible]  # Get position embeddings for visible patches
        visible_patches = visible_patches + pos_embed_visible
        
        # Process visible patches through encoder
        encoded_visible = self.transformer_encoder(visible_patches.unsqueeze(1)).squeeze(1)
        
        # DECODER APPROACH: Following the original MAE more closely
        # 1. Create full sequence of tokens with mask tokens
        # 2. Replace mask tokens with encoded visible patches at appropriate positions
        # 3. Add positional embeddings to all tokens
        # 4. Process through decoder
        
        # Create decoder input by expanding mask token to full sequence
        # For each sample-channel, create a sequence of patch_size mask tokens
        decoder_tokens = torch.zeros(num_samples, num_patches, self.hid_dim // self.patch_size, device=self.device)
        
        # Fill in encoded visible tokens
        decoder_tokens[sample_indices_visible, patch_indices_visible] = encoded_visible
        
        # Create mask for mask tokens (inverse of boolean mask)
        mask_token_mask = ~bool_mask_flat
        
        # Fill in mask tokens
        mask_token_indices = torch.nonzero(mask_token_mask)
        sample_indices_mask = mask_token_indices[:, 0]
        patch_indices_mask = mask_token_indices[:, 1]
        decoder_tokens[sample_indices_mask, patch_indices_mask] = self.mask_token.squeeze(0)
        
        # Add positional embeddings to all tokens
        decoder_tokens = decoder_tokens + self.patch_pos_embed  # Broadcasting handles this
        
        # Reshape for more efficient batch processing
        decoder_tokens_flat = decoder_tokens.reshape(-1, self.hid_dim // self.patch_size)
        
        # Process through decoder
        decoded_tokens = self.decoder(decoder_tokens_flat)
        decoded_tokens = decoded_tokens.view(num_samples, num_patches, -1)
        
        # Reshape back to (B, D, patch_size, hid_dim//patch_size)
        full_reconstruction = decoded_tokens.view(B, D, self.patch_size, self.hid_dim // self.patch_size)
        
        # Create signal projector
        signal_projector = nn.Linear(self.hid_dim, L).to(self.device)
        
        # Reshape and project all channels at once
        reconstructed_latent = full_reconstruction.reshape(B * D, self.hid_dim)
        reconstructed_signal_flat = signal_projector(reconstructed_latent)
        reconstructed_signals = reconstructed_signal_flat.view(B, D, L).transpose(1, 2)  # (B, L, D)
        
        # Compute loss ONLY on the masked patches (MAE approach)
        masked_patches_orig = orig_patches[~bool_mask_flat.view(B, D, self.patch_size)]
        masked_patches_recon = full_reconstruction[~bool_mask_flat.view(B, D, self.patch_size)]
        loss = F.mse_loss(masked_patches_recon, masked_patches_orig)
        
        return {
            'loss': loss,
            'reconstructed_signal': reconstructed_signals,
            'encoded_representation': output
        }