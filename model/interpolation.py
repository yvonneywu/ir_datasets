import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gmae_v3 import gMAE

class TSInterpolation(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super(TSInterpolation, self).__init__()
        self.config = config
        self.device = config.device
        
        # Load pretrained gMAE model
        self.base_model = gMAE(config)
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained model from {pretrained_path}")
        
        # Freeze base model parameters if specified
        if config.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Define task-specific head for interpolation
        # self.interpolation_head = nn.Sequential(
        #     nn.Linear(config.hid_dim, config.hid_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hid_dim * 2, 1)  # Output is a single value for each time point
        # )

        self.interpolation_head = nn.Sequential(
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.LayerNorm(config.hid_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.LayerNorm(config.hid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hid_dim, 1)
        )

        
    def forward(self, value, time_steps, mask, interp_mask):

        B, L, D = value.shape
    
        # Prepare input for base model
        X = value.permute(0, 2, 1).reshape(-1, L, 1)  # (B*D, L, 1)
        time_steps_expanded = time_steps.unsqueeze(-1).expand(B, L, D)
        time_steps_reshaped = time_steps_expanded.permute(0, 2, 1).reshape(-1, L, 1)
        mask_reshaped = mask.permute(0, 2, 1).reshape(-1, L, 1)
        
        # # ##TTCN#
        # # Get time embeddings
        # te_his = self.base_model.LearnableTE(time_steps_reshaped)
        # X_with_time = torch.cat([X, te_his], dim=-1)
        # latent_features = self.base_model.TTCN(X_with_time, mask_reshaped)
        # latent_features = latent_features.view(B, D, self.config.hid_dim)
        
        # # # Get latent representations from TTCN
        time_steps_expanded = time_steps.unsqueeze(1).expand(-1, D, -1)  # (B, D, L)
        time_steps_reshaped = time_steps_expanded.reshape(-1, L)          # (B*D, L)

        key = self.base_model.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        time_query = self.base_model.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        cls_query = self.base_model.learn_time_embedding(self.base_model.cls_query.unsqueeze(0).to(self.device))  # (1, 128, te_dim)

               # IMPORTANT - Use mask directly, don't concatenate it twice
        #  The mask should match the value dimensions
        mtan_out = self.base_model.time_att(
            cls_query.repeat(B*D, 1, 1),  # Repeat cls_query for each channel
            key,                          # Time embeddings as key
            X,                      # Values only
            mask_reshaped                 # Use mask directly
        )

   
        # mtan_out shape should be (B*D, 128, hidden_size)
        # Replace mean pooling with attention pooling
        attention_weights = F.softmax(torch.bmm(
            self.base_model.query_attention.repeat(B*D, 1, 1), 
            mtan_out.transpose(1, 2)
        ), dim=-1)
        output = torch.bmm(attention_weights, mtan_out).squeeze(1).view(B, D, -1)
        latent_features = output  # shape: (B, D, hid_dim)

        ###MTAN, previous version without modifying any input shape/adding layer...###
        # output = self.base_model(value, time_steps, mask)
        # latent_features = output['encoded_representation']  # shape: (B, D, hid_dim)
        
        ###### get the latent representation of the masked time points
        # Apply interpolation head to ALL latent representations at once
        all_predictions = self.interpolation_head(latent_features.reshape(-1, self.config.hid_dim))
        all_predictions = all_predictions.view(B, D, -1)  # Reshape to (B, D, output_size)
        
        # Use the interpolation mask to create full predictions tensor
        predictions = torch.zeros_like(value)
        # predictions = value.clone()  
        
        # Create a mask for where we need predictions
        mask_indices = torch.nonzero(interp_mask, as_tuple=True)
        
        # Vectorized assignment using advanced indexing
        b_indices, t_indices, d_indices = mask_indices
        predictions[b_indices, t_indices, d_indices] = all_predictions[b_indices, d_indices, 0]
        
        return predictions