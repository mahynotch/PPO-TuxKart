import torch
import torch.nn as nn
from os import path
import torch.nn.functional as F
from planner import ImageDecoder, AimPointPredictor, spatial_argmax

# spatial_argmax function (already provided in the problem description, ensure it's available)
# def spatial_argmax(logit):
#     """
#     Compute the soft-argmax of a heatmap
#     :param logit: A tensor of size BS x H x W
#     :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
#     """
#     weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
#     return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
#                         (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        # Now, the 'C' dimension is the last one.
        # nn.LayerNorm(C) or super().forward(x) (if normalized_shape was set to C in __init__)
        # will normalize across this last dimension (channels) for each (B, H, W) location.
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        return x

class PPOPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch import load # If loading pretrained parts here

        self.feature_extractor = ImageDecoder()
        # self.feature_extractor.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'decoder.th'), map_location='cuda')) # Example
        # self.feature_extractor.requires_grad_(False) # Freeze the feature extractor
        
        feature_dim = 512

        self.policy_head_heatmap = AimPointPredictor()
        # self.policy_head_heatmap.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'predictor.th'), map_location='cuda')) # Example

        self.value_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            LayerNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            LayerNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            LayerNorm2d(32),
            nn.AdaptiveAvgPool2d(1), # Reduces H, W to 1, 1
            nn.Flatten(),
            nn.Linear(32, 1)
        )


        for param in self.value_head.parameters():
            if param.dim() > 1:  # Only apply to weight matrices, not biases
                nn.init.kaiming_normal_(param)

    def forward(self, img):
        """
        Forward pass through the network.
        @img: (B,3,96,128) or (B,96,128,3)
        return: heatmap_logits (B,H,W), value (B,1)
        """
        if img.dim() == 4 and img.shape[1] != 3 and img.shape[3] == 3: # (B, H, W, C)
            img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        elif img.dim() != 4 or img.shape[1] != 3:
             raise ValueError(f"Input image has unexpected shape: {img.shape}. Expected (B, C, H, W) or (B, H, W, C) with C=3")

        features = self.feature_extractor(img) # (B, feature_dim, H_feat, W_feat)
        heatmap_logits = self.policy_head_heatmap(features) # Expected (B, H_feat, W_feat)
        value = self.value_head(features)

        return heatmap_logits, value

    def get_action(self, img: torch.Tensor, deterministic: bool = False):
        """
        Get action from the model.
        The policy is now categorical over the pixels of the heatmap.
        @img: (B,C,H,W) e.g. (B,3,96,128)
        @deterministic: If True, return the spatial_argmax action.
        return:
            action_normalized (B,2): Action in normalized (-1,1) coordinates for the environment.
            action_pixel_indices (B,): Sampled pixel indices (flattened) for PPO buffer. None if deterministic.
            log_prob (B,1): Log probability of the sampled action_pixel_indices. Zeros if deterministic.
            value (B,1): Value estimate.
        """
        heatmap_logits, value = self.forward(img)
        batch_size, H, W = heatmap_logits.shape

        if deterministic:
            action_normalized = spatial_argmax(heatmap_logits)
            action_pixel_indices = None
            log_prob = torch.zeros(batch_size, 1, device=img.device)
        else:
            # Stochastic action: sample from the categorical distribution defined by the heatmap
            probs = F.softmax(heatmap_logits.view(batch_size, -1), dim=-1) # (B, H*W)
            dist = torch.distributions.Categorical(probs=probs)
            action_pixel_indices = dist.sample() # (B,)
            log_prob = dist.log_prob(action_pixel_indices).unsqueeze(-1) # (B,1)

            h_indices = action_pixel_indices // W
            w_indices = action_pixel_indices % W

            xs = torch.linspace(-1, 1, W, device=img.device, dtype=torch.float32)
            ys = torch.linspace(-1, 1, H, device=img.device, dtype=torch.float32)

            action_sampled_x = xs[w_indices]
            action_sampled_y = ys[h_indices]

            action_normalized = torch.stack((action_sampled_x, action_sampled_y), dim=1)
            action_normalized = torch.clamp(action_normalized, -1.0, 1.0)

        return action_normalized, action_pixel_indices, log_prob, value

class PPOPlannerWithLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # 96x128 -> 48x64
            LayerNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 48x64 -> 24x32
            LayerNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24x32 -> 12x16
            LayerNorm2d(128),
            nn.ReLU(),
        )
        feature_dim = 128 # Output channels from the feature_extractor
        # The output feature map size from the above for 96x128 input will be 12x16 (H_feat x W_feat)

        # This head outputs the heatmap logits.
        # It takes features (B, 128, H_feat, W_feat) and outputs (B, 1, H_feat, W_feat)
        self.policy_head_heatmap_conv = nn.Conv2d(feature_dim, 1, kernel_size=1, stride=1, padding=0)

        # policy_head_log_std and associated min/max are removed.

        self.value_head = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, img):
        """
        Forward pass through the network.
        @img: (B,3,96,128) or (B,96,128,3)
        return: heatmap_logits (B,H_feat,W_feat), value (B,1)
        """
        if img.dim() == 4 and img.shape[1] != 3 and img.shape[3] == 3:
            img = img.permute(0, 3, 1, 2)
        elif img.dim() != 4 or img.shape[1] != 3 :
             raise ValueError(f"Input image has unexpected shape: {img.shape}. Expected (B, C, H, W) or (B, H, W, C) with C=3")

        features = self.feature_extractor(img) # (B, feature_dim, H_feat, W_feat)
        
        # policy_head_heatmap_conv outputs (B, 1, H_feat, W_feat)
        # We need to squeeze the channel dimension to get (B, H_feat, W_feat)
        heatmap_logits = self.policy_head_heatmap_conv(features).squeeze(1)
        
        value = self.value_head(features)
        
        return heatmap_logits, value

    # The get_action method can be identical to the one in PPOPlanner
    # as it relies on the output signature of forward() being (heatmap_logits, value)
    def get_action(self, img: torch.Tensor, deterministic: bool = False):
        """
        Get action from the model.
        The policy is now categorical over the pixels of the heatmap.
        @img: (B,C,H,W) e.g. (B,3,96,128)
        @deterministic: If True, return the spatial_argmax action.
        return:
            action_normalized (B,2): Action in normalized (-1,1) coordinates for the environment.
            action_pixel_indices (B,): Sampled pixel indices (flattened) for PPO buffer. None if deterministic.
            log_prob (B,1): Log probability of the sampled action_pixel_indices. Zeros if deterministic.
            value (B,1): Value estimate.
        """
        heatmap_logits, value = self.forward(img)
        batch_size, H, W = heatmap_logits.shape

        if deterministic:
            action_normalized = spatial_argmax(heatmap_logits)
            action_pixel_indices = None
            log_prob = torch.zeros(batch_size, 1, device=img.device)
        else:
            probs = F.softmax(heatmap_logits.view(batch_size, -1), dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action_pixel_indices = dist.sample()
            log_prob = dist.log_prob(action_pixel_indices).unsqueeze(-1)

            h_indices = action_pixel_indices // W
            w_indices = action_pixel_indices % W

            xs = torch.linspace(-1, 1, W, device=img.device, dtype=torch.float32)
            ys = torch.linspace(-1, 1, H, device=img.device, dtype=torch.float32)

            action_sampled_x = xs[w_indices]
            action_sampled_y = ys[h_indices]

            action_normalized = torch.stack((action_sampled_x, action_sampled_y), dim=1)
            action_normalized = torch.clamp(action_normalized, -1.0, 1.0)
            
        return action_normalized, action_pixel_indices, log_prob, value

def save_model(model: nn.Module, model_filename: str = 'ppo_planner.th'):
    from torch import save
    try:
        # Handle case where __file__ might not be defined (e.g. in some notebooks)
        base_path = path.dirname(path.abspath(__file__)) if '__file__' in globals() else '.'
        save_path = path.join(base_path, model_filename)
    except NameError: # Fallback for environments where __file__ is not defined
        save_path = model_filename


    if isinstance(model, (PPOPlanner, PPOPlannerWithLayerNorm)):
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path) # Use torch.save
    else:
        raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(model_filename: str = 'ppo_planner.th', device: str = 'cpu', use_layernorm_version=False) -> nn.Module:
    from torch import load
    try:
        base_path = path.dirname(path.abspath(__file__)) if '__file__' in globals() else '.'
        load_path = path.join(base_path, model_filename)
    except NameError:
        load_path = model_filename
        
    print(f"Loading model from {load_path} to {device}")
    
    if use_layernorm_version:
        r = PPOPlannerWithLayerNorm()
    else:
        r = PPOPlanner() 
    
    r.load_state_dict(load(load_path, map_location=device))
    r.to(device) # Ensure model is moved to the specified device
    r.eval() 
    return r