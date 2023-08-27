import torch
from torch import nn
import utils


class Seq2Seq(nn.Module):
    def __init__(self, prior, post, shift, decoder, latent_dim, window, N, dropout):
        super().__init__()
        self.prior = prior
        self.post = post
        self.shift = shift
        self.decoder = decoder
        window1 = int(window / 2)
        # 计算 S 均值
        self.dec_mean = nn.Sequential(
            nn.Linear(decoder.hid_dim, window),
            nn.GELU(),
            nn.Linear(window, window),
            nn.GELU(),
            nn.Linear(window, window),

        ).cuda()
        # 计算 S 方差
        self.dec_std = nn.Sequential(
            nn.Linear(decoder.hid_dim, window),
            nn.GELU(),
            nn.Linear(window, window),
            nn.Softplus(),
            nn.Linear(window, window),
        ).cuda()
        self.gen = nn.Sequential(
            nn.Linear(window, window),
            nn.GELU(),
            nn.Linear(window, window),
            nn.GELU(),
            nn.Linear(window, window),
            nn.GELU(),
            # nn.Linear(window, window),
            nn.Linear(window, window),
        ).cuda()
        self.N = N
        dim = 1
        self.self_att = utils.SelfAttention(dim_in=dim, dim_k=dim, dim_v=dim).cuda()
        self.latent_dim = latent_dim
        self.window = window
        self.a = 1
        self.pe = utils.PositionalEncoding(d_model=1, dropout=0, max_len=window)


    def forward(self, x_data, y, device):
        # history_z(batch, seq_len, input_size)
        history_z, prior_list = self.prior(x_data)
        post_list = self.post(x_data, y)
        # 计算时间窗口 KL 之和
        kl = utils.get_kl(post_list, prior_list)
        # out_put(batch_size, emd_size)
        hidden = self.shift(history_z)
        Z = self.decoder(history_z[:, -1, :], hidden, device)
        # 生成 S 的
        mu = self.dec_mean(Z)
        std = self.dec_std(Z)
        g_result = 0
        for i in range(self.N):
            g_result = g_result + utils.X_Repasampling(mu, std, device)
        g_result = g_result / self.N
        # 位置编码
        position_emd = self.pe.pe[:, :].cuda()
        last_x = g_result.unsqueeze(1)
        input = (last_x + position_emd * last_x).permute(0, 2, 1)
        att = self.self_att(input).squeeze()
        result = self.gen(att)
        return result, kl


    def predict(self, x_data, device):
        # GRU_input(batch, seq_len, input_size)
        GRU_input = self.prior.predict(x_data)
        # out_put(batch_size, emd_size)
        hidden = self.shift(GRU_input)
        # 获取预测隐藏状态
        Z = self.decoder(GRU_input[:, -1, :], hidden, device)
        mu = self.dec_mean(Z)
        std = self.dec_std(Z)
        g_result = 0
        for i in range(self.N):
            g_result = g_result + utils.X_Repasampling(mu, std, device)
        g_result = g_result / self.N
        # 位置编码
        position_emd = self.pe.pe[:, :].cuda()
        last_x = g_result.unsqueeze(1)
        input = (last_x + position_emd * last_x).permute(0, 2, 1)
        att = self.self_att(input).squeeze()
        result = self.gen(att)
        return result