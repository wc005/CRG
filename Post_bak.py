from torch import nn
import torch
import numpy as np
from torch.distributions import Normal


class Post(nn.Module):
    def __init__(self, z_size, hidden_size, n_layers, dropout):
        super(Post, self).__init__()
        self.z_size = z_size
        self.rnn_mu = nn.GRU(z_size, hidden_size, n_layers, batch_first=True, dropout=dropout).cuda()
        self.rnn_std = nn.GRU(z_size, hidden_size, n_layers, batch_first=True, dropout=dropout).cuda()
        self.get_post_mu = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, z_size),
            nn.Softplus()
        ).cuda()
        self.get_post_std = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, z_size),
            nn.Softplus()
        ).cuda()


    def forward(self, history, future):
        # history: [batch size, sequence len,  z_size]
        # future: [batch size, z_size]
        post_list = []
        prior_list = []
        for i in range(history.shape[1]):
            h = history[:, i, :]
            Z = torch.stack((h, future), dim=1)
            # 计算均值
            outputs, hidden = self.rnn_mu(Z[:, :, :self.z_size])
            post_mu = self.get_post_mu(outputs[:, -1:].squeeze())
            # 计算方差
            outputs, hidden = self.rnn_std(Z[:, :, self.z_size:])
            post_std = self.get_post_std(outputs[:, -1:].squeeze())
            post_list.append(Normal(post_mu, post_std))
            prior_list.append(Normal(h[:, :self.z_size], h[:, self.z_size:]))
        return prior_list, post_list