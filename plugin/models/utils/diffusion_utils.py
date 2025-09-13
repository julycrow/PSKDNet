import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t.
    
    Args:
        a: Tensor containing coefficients for all timesteps
        t: Timestep to extract
        x_shape: Shape of x
        
    Returns:
        Tensor of coefficients for timestep t
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def get_ddpm_schedule(beta_schedule='linear', timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Get DDPM scheduler parameters.
    
    Args:
        beta_schedule: Type of beta schedule
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tensor of beta values
    """
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps)
    elif beta_schedule == 'cosine':
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")

def add_noise(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Add noise to data for diffusion process.
    
    Args:
        x_start: Starting clean data
        t: Timestep
        sqrt_alphas_cumprod: Precomputed sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1-alphas_cumprod)
        
    Returns:
        Noisy data
    """
    noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionPredictor(nn.Module):
    """DiffusionDet predictor module for map element detection."""
    def __init__(self, 
                 in_channels, 
                 hidden_dim=256, 
                 num_classes=3, 
                 num_queries=100, 
                 num_points=20, 
                 coord_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        # Time embedding
        time_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Spatial feature encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Point feature processor
        point_dim = num_points * coord_dim + 1 + num_classes  # coordinates + score + class
        self.point_processor = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Cross-attention for feature fusion
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 8, batch_first=True)
            
        # Output denoiser
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, point_dim)
        )
        
    def forward(self, x, spatial_features, t):
        """Forward pass of diffusion predictor.
        
        Args:
            x: Input points with noise [B, num_queries, num_points*coord_dim+1+num_classes]
            spatial_features: BEV features [B, C, H, W]
            t: Current timestep [B]
            
        Returns:
            Predicted denoised points
        """
        # Get batch size
        bs = x.shape[0]
        
        # Process time embeddings
        t_emb = self.time_mlp(t)  # [B, time_dim]
        
        # Process spatial features
        s_feats = self.spatial_encoder(spatial_features)  # [B, hidden_dim, H, W]
        h, w = s_feats.shape[2:]
        s_feats = s_feats.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # Process point features
        p_feats = self.point_processor(x)  # [B, num_queries, hidden_dim]
        
        # Cross-attention between point features and spatial features
        p_feats = p_feats + self.cross_attention(
            query=p_feats,
            key=s_feats,
            value=s_feats,
            need_weights=False
        )[0]
        
        # Combine with time embedding and predict denoised points
        t_emb = t_emb.unsqueeze(1).repeat(1, self.num_queries, 1)  # [B, num_queries, time_dim]
        pred = self.denoiser(torch.cat([p_feats, t_emb], dim=-1))  # [B, num_queries, point_dim]
        
        return pred