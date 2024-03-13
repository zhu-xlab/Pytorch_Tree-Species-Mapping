import torch
import torch.nn as nn
import torch.nn.functional as F


class ViViT(nn.Module):
    def __init__(self, config):
        super(ViViT, self).__init__()
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        # self.spatial_dim = config.spatial_dim
        self.temporal_dim = config.num_length
        self.dropout = config.dropout

        # Spatial embedding layer
        # self.spatial_embedding = nn.Linear(config.input_channels * config.w * config.h, self.embedding_dim)
        self.spatial_embedding = nn.Linear(config.input_channels, self.embedding_dim)

        # Temporal positional encoding
        self.temporal_positions = nn.Parameter(torch.randn(self.temporal_dim, self.embedding_dim))

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear(self.embedding_dim, config.num_classes)

    def forward(self, x):
        batch_size, seq_len, input_channels, height, width = x.size()

        # x = x.permute(1, 0, 2, 3, 4)
        # x = x.view(seq_len, batch_size, input_channels * height * width)
        x = x.permute(1, 0, 3, 4, 2)
        # x = x.view(seq_len, batch_size*height*width, input_channels)

        # Spatial embedding
        x = self.spatial_embedding(x)
        x = x.view(seq_len, batch_size * height * width, self.embedding_dim)

        # Add temporal positional encoding
        # temporal_pos_enc = self.temporal_positions[:seq_len].unsqueeze(1)
        # temporal_pos_enc = temporal_pos_enc.repeat(1, batch_size * height * width, 1)
        # x = x + temporal_pos_enc

        # Reshape for Transformer input
        # x = x.view(batch_size * seq_len, height * width, self.embedding_dim)
        # x = x.permute(1, 0, 2)  # [height*width, batch_size*seq_len, embedding_dim]

        # Transformer encoding
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Reshape back to sequence format
        # x = x.permute(1, 0, 2).contiguous().view(batch_size, seq_len, -1)
        x = x.permute(1, 0, 2).contiguous().view(batch_size, height, width, seq_len, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.view(batch_size, seq_len*self.embedding_dim, height, width)
        # print(x.shape)

        # # Global average pooling
        x = F.avg_pool2d(x, (height, width))
        x = x.view(batch_size, seq_len, -1)

        # Classification
        x = self.fc(x.mean(dim=1))

        return x

