from torch import nn


class Shift(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout).cuda()


    def forward(self, input):
        # input: [batch size, sequence len,  input_size]
        # outputs: [batch size, sequence len, hid_dim * directions]
        # hidden: [n_layers, batch size, hid_dim]
        outputs, hidden = self.rnn(input)
        # s: [num_layers, batch size, hid dim]
        hidden = hidden[self.n_layers-1].squeeze()
        return hidden
