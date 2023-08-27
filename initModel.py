from Blocks import Decoder, Post, Prior, Shift, Seq2Seq


def initCRG(window, args):
    '''
    初始化模型
    :param window:
    :param args:
    :return:
    '''
    # 编码器 参数
    input_size = window
    encoder_input_dim = 2 * args.z_size
    encoder_hid_dim = args.hidden_size
    decoder_hidden_dim = encoder_hid_dim
    # 先验 Z 概率
    prior = Prior.autoencoder(input_size, args.hidden_size, args.z_size)
    # 后验 Z 概率
    post = Post.Post(input_size, args.hidden_size, args.z_size, args.n_layers, args.dropout)
    # RNN 漂移网络
    ShiftModel = Shift.Shift(encoder_input_dim, encoder_hid_dim, args.n_layers, args.dropout)
    # 解码器
    decoder = Decoder.Decoder(args.z_size, decoder_hidden_dim, args.z_samples)
    model = Seq2Seq.Seq2Seq(prior, post, ShiftModel, decoder, args.z_size, window, args.s_samples, args.dropout)
    return model