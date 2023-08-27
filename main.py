import torch
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import data_process, utils
import pickle
from itertools import product
from dataclass import ETT, exchange_rate, national_illness, traffic, weather, electricity, wpf_data_ID1
import initModel
import argparse
from mains.busi import train, valid, test
import tools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRG for Time Series Forecasting')
    # 超参数设置
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--data_set', type=str, default='wpf_data_ID1', help='batch size of train input data')
    parser.add_argument('--shift_len', type=int, default=3, help='shift length of laten Z')
    # (420, 400), (220, 200)
    # (62, 60), (50, 48), (38, 36), (26, 24)
    parser.add_argument('--step_list', type=list, default=[(820, 800)],
                        help='(window, step),the length of the window and step')
    parser.add_argument('--hidden_size', type=int, default=96, help='(window, step),the length of the window and step')
    parser.add_argument('--n_layers', type=int, default=2, help='the num of layers for GRU')
    parser.add_argument('--z_size', type=int, default=128, help='the dim of laten Z')
    parser.add_argument('--balance_factor', type=float, default=1000, help='the dim of laten Z')
    parser.add_argument('--s_samples', type=int, default=10, help='the times of sample follow the distribution of laten Z')
    parser.add_argument('--z_samples', type=int, default=10, help='the times of sample follow the distribution S')
    parser.add_argument('--epoch', type=int, default=2000, help='the times of training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--dropout', type=float, default=0.01, help='')
    parser.add_argument('--stop_f', type=float, default=0.0002, help='')

    args = parser.parse_args()
    # 停止训练误差降低阈值
    stop_f = 0.00000001
    dataname = args.data_set
    path = "data/{}/{}.csv".format(dataname, dataname)
    # 测试结果存储结构
    results_class = wpf_data_ID1()
    for steps, item in product(args.step_list, results_class.item_list):
        window, step = steps
        print('--------------step:{},item:{}----------------'.format(step, item))
        CRG = initModel.initCRG(window, args)
        CRG_best = initModel.initCRG(window, args)
        # 打印参数数量
        settings = utils.print_model_parameters(CRG, args)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(CRG.parameters(), lr=args.learning_rate)
        Scaler = StandardScaler()
        dataset = data_process.get_data(path, item, window, step, Scaler)
        trainset, valset, testset = data_process.getset(dataset, args.shift_len, args.batch_size)
        best_score = float('inf')
        stop_flag = False
        draw_tools = tools.tools(1, item, step, args.data_set)
        train_mse_list = []
        valid_mse_list = []
        for i in tqdm(range(args.epoch)):
            if stop_flag:
                break
            # 训练
            train_mse = train(CRG, trainset, loss_function, optimizer, device, step, i, args.balance_factor, args, train_mse_list)
            # 验证
            if i % 100 == 0:
                CRG.eval()
                best_score, stop_flag = valid(CRG, valset, device, window, step, loss_function, best_score,
                                                dataname, item, stop_f, stop_flag, args, valid_mse_list)
                # 每训练 100 次减小学习率
                if i % 100 == 0:
                    for param_group in optimizer.param_groups:
                        print(param_group['lr'])
                        param_group['lr'] = param_group['lr'] - param_group['lr'] / 2
            # 测试
            if True == stop_flag or 0 == (i + 1) % args.epoch or 0 == (i + 1) % 400:
                test(CRG_best, testset, device, dataname, item, step, window, loss_function, results_class, settings,
                     draw_tools)
        path_loss = 'loss/{}_loss_{}'.format(args.data_set, step)
        utils.saveLoss(path_loss, train_mse_list, valid_mse_list)
        path_pic = 'loss/{}_loss_{}.pdf'.format(args.data_set, step)
        tools.drawloss(path_pic, train_mse_list, valid_mse_list)
        # 每一预测步长结束后，存储
        with open("results/{}_{}.pkl".format(dataname, step), "wb") as tf:
            pickle.dump(results_class.finalreuslts_dic, tf)

    with open("results/{}_final.pkl".format(dataname), "wb") as tf:
        pickle.dump(results_class.finalreuslts_dic, tf)
