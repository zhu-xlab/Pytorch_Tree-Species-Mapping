import torch
import torch.nn as nn
import math


class ViT(nn.Module):
    def __init__(self, config, device):
        super(ViT, self).__init__()
        self.time_steps = config.num_length
        self.input_channels = config.input_channels
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout
        self.num_classes = config.num_classes
        self.patch_size = config.patch_size
        self.batch_size = config.batch_size
        self.w = config.w
        self.h = config.h
        self.device = device

        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(self.input_channels*self.time_steps, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout
            ),
            num_layers=config.num_layers
        )

        # Classification Head
        self.fc = nn.Linear(self.w * self.h * self.embedding_dim, self.num_classes)

    def generate_positional_encoding(self, batch_size, time_steps, embedding_dim):
        # Generate positional encoding for the given sequence length and embedding dimension
        positional_encoding = torch.zeros(time_steps, embedding_dim)
        position = torch.arange(0, time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(positional_encoding.unsqueeze(0).repeat(batch_size, 1, 1)).to(self.device)

    def forward(self, x):
        batch_size, time_steps, input_channels, height, width = x.size()

        # Rearrange input data to batch_size x (time_steps * input_channels) x height x width
        x = x.view(batch_size, time_steps * input_channels, height, width)

        # Patch Embedding
        x = self.patch_embedding(x)  # Output shape: batch_size x embedding_dim x H' x W'
        x = x.permute(0, 2, 3, 1)  # Rearrange dimensions for Transformer input

        B, H, W, C = x.shape

        # Flatten the spatial dimensions
        x = x.view(B, H * W, C)

        # Add positional encoding
        x = x + self.generate_positional_encoding(B, H * W, C)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Reshape for classification
        x = x.view(B, -1)  # Output shape: batch_size x (time_steps * embedding_dim)

        # Classification Head
        x = self.fc(x)

        return x

