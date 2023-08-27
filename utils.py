import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.distributions import kl_divergence


def get_kl(q, p):
    kl = 0
    for i in range(len(q)):
        kl = kl + kl_divergence(q[i], p[i]).mean()
    return kl


def get_eign(x, t, device):
    batch, hid_emd = x.shape
    eigen = torch.zeros((batch, t*2)).to(device)
    for i in range(t):
        # 转为弧度
        radian = i * x * torch.pi / 180
        re = torch.mean(torch.cos(radian), axis=1)
        eigen[:, i] = re
        im = torch.mean(torch.sin(radian), axis=1)
        eigen[:, i+1] = im
    return eigen


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(-max_len, max_len, step=2, dtype=torch.float).unsqueeze(1)
        # temp = torch.arange(0, d_model, 2).float()
        # div_term = torch.exp(temp * (-math.log(10000.0) / d_model))
        # odd_emd = torch.sin(position * div_term)
        # even_emd = torch.cos(position * div_term)
        odd_emd = torch.sin(position/max_len)
        pe[:, ] = odd_emd
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            # >>> output = pos_encoder(x)
        """
        b = self.pe[:x.size(0), :]
        return b


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=True).cuda()
        self.linear_k = nn.Linear(dim_in, dim_k, bias=True).cuda()
        self.linear_v = nn.Linear(dim_in, dim_v, bias=True).cuda()
        self._norm_fact = 1 / math.sqrt(dim_k)


    def forward(self, x):
        # x: batch, n, dim_in
        batch, window, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        mask = 1 - torch.triu(torch.ones((window, window), dtype=torch.uint8), diagonal=1).cuda()
        mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, L, L]
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = dist.masked_fill(mask == 0, -1e9)
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att



# 重参数采样
def Z_Repasampling(mu, var2, device):
    mean = torch.zeros(mu.shape, device=device)
    # print(mean.grad)
    var = torch.ones(var2.shape, device=device)
    # print(mean.grad)
    shape = mean.size()
    epsilon = torch.cuda.FloatTensor(shape)
    torch.normal(mean, var, out=epsilon)
    sample = mu + epsilon * torch.sqrt(var2)
    return sample

# 重参数采样
def X_Repasampling(mu, logvar2, device):
    mean = torch.zeros(mu.shape, device=device)
    # print(mean.grad)
    var = torch.ones(logvar2.shape, device=device)
    # print(mean.grad)
    shape = mean.size()
    epsilon = torch.cuda.FloatTensor(shape)
    torch.normal(mean, var, out=epsilon)
    sample = mu + epsilon * torch.sqrt(torch.exp(logvar2))
    # sample = mu + epsilon * torch.sqrt(logvar2)
    return sample


def MAPE(y, y_pred):
    mape = torch.abs((y - y_pred) / y)
    mape[torch.isinf(mape)] = 0.5
    result = torch.mean(mape)
    return result * 100


def MAE(y, y_pred):
    mae = torch.abs(y - y_pred)
    result = torch.mean(mae)
    return result


def print_model_parameters(model, args, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    settings = '{}_{}_{}_{}_hs{}_nl{}_zs{}_ssam{}_zsam{}_bf{}_ep{}_lr{}_dr{}_sf{}'.format(
        args.batch_size,
        args.data_set,
        args.shift_len,
        str(args.step_list),
        args.hidden_size,
        args.n_layers,
        args.z_size,
        args.s_samples,
        args.z_samples,
        args.balance_factor,
        args.epoch,
        args.learning_rate,
        args.dropout,
        args.stop_f)
    return settings

def saveResult(setting, result):
    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write('result:{}'.format(result))
    f.write('\n')
    f.write('\n')
    f.close()

def saveLoss(path, train_mse_list, valid_mse_list):
    train_loss = np.array(train_mse_list)
    valid_loss = np.array(valid_mse_list)
    f = open(path, 'w')
    f.write('train_loss:{}'.format(train_loss))
    f.write('valid_loss:{}'.format(valid_loss))
    f.write('\n')
    f.write('\n')
    f.close()

def saveTime(path, time):
    time = np.mean(np.array(time))
    f = open(path, 'a')
    f.write('time:{}'.format(time))
    f.write('\n')
    f.write('\n')
    f.close()