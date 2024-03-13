import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, _ = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class CustomTransformerModel(nn.Module):
    def __init__(self, config):
        super(CustomTransformerModel, self).__init__()
        self.num_classes = config.num_classes
        self.num_channels = config.num_channels
        self.w = config.w
        self.h = config.h
        self.size = config.w * config.h
        self.num_layers = config.num_layers
        self.forward_expansion = config.forward_expansion

        self.channel_transformer = Transformer(embed_size=25, heads=5, num_layers=self.num_layers, dropout=0.5, forward_expansion=self.forward_expansion)
        self.spatial_transformer = Transformer(embed_size=120, heads=5, num_layers=self.num_layers, dropout=0.5, forward_expansion=self.forward_expansion)

        # FC layers
        self.fc1 = nn.Linear(self.size * self.num_channels + self.num_channels, 2048)  # Combined output of both branches
        self.fc2 = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        batch_size, input_channels, height, width = x.size()

        # Channel Attention branch
        channel_x = x.view(batch_size, self.num_channels, -1).permute(1, 0, 2)  # (64, 120, 25)
        channel_out = self.channel_transformer(channel_x)

        # Spatial Attention branch
        spatial_outs = []
        for h in range(self.w):
            for w in range(self.h):
                spatial_x = x[:, :, h, w].unsqueeze(2).permute(2, 0, 1)
                spatial_out = self.spatial_transformer(spatial_x)
                spatial_outs.append(spatial_out)

        spatial_out = torch.mean(torch.stack(spatial_outs, dim=0), dim=0).squeeze(0)

        # Flatten outputs
        channel_out = channel_out.permute(1, 0, 2).contiguous().view(batch_size, -1)
        spatial_out = spatial_out.view(batch_size, -1)

        # Concatenate outputs
        combined_out = torch.cat([channel_out, spatial_out], dim=1)

        # Classification
        x = F.relu(self.fc1(combined_out))
        # x1 = F.normalize(x)
        x = self.fc2(x)

        return x

