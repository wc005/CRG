from torch import nn
from torch.distributions import Normal
import utils

class Decoder(nn.Module):
    def __init__(self, z_size, hidden_size,  z_samples):
        super().__init__()
        self.hid_dim = hidden_size
        self.z_samples = z_samples
        self.input_size = z_size
        self.GRUCell = nn.GRUCell(z_size*2, hidden_size).cuda()
        # self.LSTM = nn.LSTM(z_size*2, hidden_size, batch_first=True)
        # 计算Z均值
        self.mu = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),

        ).cuda()
        # 计算Z方差
        self.logvar2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),

        ).cuda()

    def forward(self, last_z, hidden, device):
        h = self.GRUCell(last_z, hidden)
        # 生成预测隐变量 Z 的分布
        mu = self.mu(h)
        logvar2 = self.logvar2(h)
        # 重参数采样获取 Z
        Z = 0
        for i in range(self.z_samples):
            Z = Z + utils.Z_Repasampling(mu, logvar2, device)
        return Z / self.z_samples

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)


    def forward(self, input):
        memory = self.transformer_encoder(input)
        return memory