import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, config):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = config.hidden_channels
        self.kernel_size = config.kernel_size
        self.padding = config.kernel_size // 2
        self.conv = nn.Conv2d(config.input_channels + config.hidden_channels, 4 * config.hidden_channels, config.kernel_size, padding=self.padding)
        self.fc = nn.Linear(config.hidden_channels, config.num_classes)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        h_t = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)

        for t in range(seq_len):
            x_t = x[:, t]
            combined = torch.cat((x_t, h_t), dim=1)
            gates = self.conv(combined)
            i_t, f_t, g_t, o_t = torch.split(gates, self.hidden_channels, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        out = h_t.mean(dim=[2, 3])
        out = self.fc(out)
        return out
