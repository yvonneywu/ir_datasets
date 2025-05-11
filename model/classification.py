import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gmae_v3 import gMAE

class TSClassification(nn.Module):
    def __init__(self, config, pretrained_path=None, num_classes=2):
        super(TSClassification, self).__init__()
        self.config = config
        self.device = config.device
        self.num_classes = num_classes
        
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
        
        # Define task-specific head for classification
        self.classification_head = nn.Sequential(
            # First aggregate across channels with attention
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.LayerNorm(config.hid_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            # Then project to classification space
            nn.Linear(config.hid_dim, config.hid_dim // 2),
            nn.LayerNorm(config.hid_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hid_dim // 2, num_classes)
        )
        
        # Channel attention for weighting different channels
        self.channel_attention = nn.Sequential(
            nn.Linear(config.hid_dim, config.hid_dim // 2),
            nn.GELU(),
            nn.Linear(config.hid_dim // 2, 1)
        )
        
    def forward(self, value, time_steps, mask):
        B, L, D = value.shape
    
        # Prepare input for base model (same as interpolation)
        X = value.permute(0, 2, 1).reshape(-1, L, 1)  # (B*D, L, 1)
        time_steps_expanded = time_steps.unsqueeze(1).expand(-1, D, -1)  # (B, D, L)
        time_steps_reshaped = time_steps_expanded.reshape(-1, L)          # (B*D, L)
        mask_reshaped = mask.permute(0, 2, 1).reshape(-1, L, 1)
        
        # Use attention-based time encoding
        key = self.base_model.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        time_query = self.base_model.learn_time_embedding(time_steps_reshaped.unsqueeze(-1))  # (B*D, L, te_dim)
        cls_query = self.base_model.learn_time_embedding(self.base_model.cls_query.unsqueeze(0).to(self.device))  # (1, 128, te_dim)

        # Process through time attention
        mtan_out = self.base_model.time_att(
            cls_query.repeat(B*D, 1, 1),  # Repeat cls_query for each channel
            key,                          # Time embeddings as key
            X,                            # Values only
            mask_reshaped                 # Use mask directly
        )
   
        # Use attention pooling
        attention_weights = F.softmax(torch.bmm(
            self.base_model.query_attention.repeat(B*D, 1, 1), 
            mtan_out.transpose(1, 2)
        ), dim=-1)
        output = torch.bmm(attention_weights, mtan_out).squeeze(1).view(B, D, -1)
        latent_features = output  # shape: (B, D, hid_dim)
        
        # Apply channel attention to weight the importance of each channel
        channel_weights = F.softmax(self.channel_attention(latent_features).squeeze(-1), dim=-1)  # (B, D)
        
        # Weighted sum of channel representations
        weighted_representation = torch.bmm(
            channel_weights.unsqueeze(1),  # (B, 1, D)
            latent_features                # (B, D, hid_dim)
        ).squeeze(1)                       # (B, hid_dim)
        
        # Apply classification head
        logits = self.classification_head(weighted_representation)  # (B, num_classes)
        
        return {
            "logits": logits,
            "channel_weights": channel_weights
        }