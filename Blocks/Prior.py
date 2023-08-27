from torch import nn
import torch
from torch.distributions import Normal


class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_size):
        super(autoencoder, self).__init__()
        self.prior_encoder_mu = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()
        self.prior_encoder_std = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()


    def forward(self, x):
        # 计算历史滑动窗口内隐变量分布
        list = []
        list_prior = []
        for i in range(0, x.shape[1]):
            X = x[:, i, :]
            # raw = x[:, i, :]
            # X = torch.as_tensor(raw, dtype=torch.float32).cuda()
            # X = torch.tensor(raw).to(torch.float32).cuda()
            prior_mu = self.prior_encoder_mu(X)
            prior_logvar = self.prior_encoder_std(X)
            Z = torch.cat((prior_mu, prior_logvar), dim=1)
            list_prior.append(Normal(prior_mu, prior_logvar))
            list.append(Z)
        # ( batch_size, seq_len, emding_size)
        history_z = torch.stack(list).permute(1, 0, 2)
        return history_z, list_prior


    def predict(self, x):
        # 计算历史滑动窗口内隐变量分布
        list = []
        for i in range(0, x.shape[1]):
            raw = x[:, i, :]
            X = torch.as_tensor(raw, dtype=torch.float32).cuda()
            # X = torch.tensor(raw).to(torch.float32).cuda()
            prior_mu = self.prior_encoder_mu(X)
            prior_logvar = self.prior_encoder_std(X)
            Z = torch.cat((prior_mu, prior_logvar), dim=1)
            list.append(Z)
        # ( batch_size, seq_len, emding_size)
        history_z = torch.stack(list).permute(1, 0, 2).to(torch.float32).cuda()
        return history_z

