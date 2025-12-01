"""
Diffusion model components for CSDI.

Contains the neural network architecture for the score-based diffusion model,
including DiffusionEmbedding and ResidualBlock components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    """Create a standard PyTorch Transformer encoder.
    
    Args:
        heads: Number of attention heads
        layers: Number of encoder layers
        channels: Model dimension
        
    Returns:
        TransformerEncoder module
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    """Create a Linear Attention Transformer.
    
    Args:
        heads: Number of attention heads
        layers: Number of layers
        channels: Model dimension
        localheads: Number of local attention heads (unused)
        localwindow: Local attention window size (unused)
        
    Returns:
        LinearAttentionTransformer module
    """
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Create a Conv1d layer with Kaiming initialization.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        
    Returns:
        Initialized Conv1d layer
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Embedding module for diffusion time steps.
    
    Creates sinusoidal embeddings for diffusion time steps and projects
    them through two linear layers with SiLU activation.
    """

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        """Initialize the diffusion embedding module.
        
        Args:
            num_steps: Number of diffusion steps
            embedding_dim: Dimension of sinusoidal embedding
            projection_dim: Dimension of projected embedding (default: embedding_dim)
        """
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        """Forward pass to get embedding for diffusion step.
        
        Args:
            diffusion_step: Tensor of diffusion step indices
            
        Returns:
            Projected embedding tensor
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        """Build sinusoidal embedding table.
        
        Args:
            num_steps: Number of diffusion steps
            dim: Half of the embedding dimension
            
        Returns:
            Embedding table tensor
        """
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    """Main diffusion model for CSDI.
    
    This module implements the score network that predicts noise
    in the reverse diffusion process.
    """

    def __init__(self, config, inputdim=2):
        """Initialize the CSDI diffusion model.
        
        Args:
            config: Configuration dictionary with model parameters
            inputdim: Input dimension (1 for unconditional, 2 for conditional)
        """
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        """Forward pass of the diffusion model.
        
        Args:
            x: Input tensor of shape (B, inputdim, K, L)
            cond_info: Conditional information tensor
            diffusion_step: Current diffusion step
            
        Returns:
            Predicted noise tensor of shape (B, K, L)
        """
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    """Residual block for the CSDI diffusion model.
    
    Combines temporal and feature attention with conditional information
    and diffusion step embedding.
    """

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        """Initialize the residual block.
        
        Args:
            side_dim: Dimension of side information
            channels: Number of channels
            diffusion_embedding_dim: Dimension of diffusion embedding
            nheads: Number of attention heads
            is_linear: Whether to use linear attention
        """
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """Apply temporal attention.
        
        Args:
            y: Input tensor
            base_shape: Original shape (B, channel, K, L)
            
        Returns:
            Tensor after temporal attention
        """
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """Apply feature attention.
        
        Args:
            y: Input tensor
            base_shape: Original shape (B, channel, K, L)
            
        Returns:
            Tensor after feature attention
        """
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        """Forward pass of the residual block.
        
        Args:
            x: Input tensor of shape (B, channel, K, L)
            cond_info: Conditional information tensor
            diffusion_emb: Diffusion step embedding
            
        Returns:
            Tuple of (output tensor, skip connection tensor)
        """
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
