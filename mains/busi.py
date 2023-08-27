import torch
from tqdm.auto import tqdm
import numpy as np
import data_process, utils, tools
import time
from torch.nn.utils import clip_grad_norm_


def train(CRG, trainset, loss_function, optimizer, device, step, i, balance_factor, args, train_mse_list):
    loss_mse = []
    loss_mae = []
    loss_mape = []
    back_time_list = []
    mse_train = 0

    for j, da in enumerate(trainset):
        time1 = time.time()
        x, y = da
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
        y = torch.as_tensor(y, dtype=torch.float32).squeeze().cuda()
        if 1 == len(x):
            continue
        CRG.zero_grad()
        pre_y, kl_loss = CRG(x, y, device)
        mse_loss = loss_function(pre_y, y)
        mae = utils.MAE(pre_y, y)
        mape = utils.MAPE(y, pre_y)
        # loss = mse_loss + kl_loss * mse_loss.item() * balance_factor
        loss = mse_loss + kl_loss * balance_factor
        assert torch.isnan(loss).sum() == 0, print(loss)
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        clip_grad_norm_(CRG.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        time2 = time.time()
        back_time_list.append(time2 - time1)
    # 记录后向过程运行时间，每个epoch记录一次
    path = './time_result/train_{}_{}_{}'.format(args.data_set, args.step_list[0][0], args.batch_size)
    utils.saveTime(path, back_time_list)

    if i % 100 == 0:
        loss_mse.append(mse_loss.item())
        loss_mae.append(mae.item())
        loss_mape.append(mape.item())
        mse_train = np.mean(np.array(loss_mse))
        mae_train = np.mean(np.array(loss_mae))
        mape_train = np.mean(np.array(loss_mape))
        print('Train***:step:{}, mse:{},mae:{},mape:{}'.format(step, mse_train, mae_train, mape_train))
        train_mse_list.append(mse_train)
    del loss_mse[:]
    del loss_mae[:]
    del loss_mape[:]


def valid(CRG, valset, device, window, step, loss_function, best_score, dataname, item, stop_f, stop_flag, args, valid_mse_list):
    with torch.no_grad():
        val_mse_list = []
        val_mae_list = []
        val_mape_list = []
        # 存储每一个 batch 测试结果
        forward_time_list = []

        for c, da in enumerate(valset):
            time1 = time.time()
            x_val, y_val = da
            if 1 == len(x_val):
                print("val_len: {}".format(len(x_val)))
                continue
            result = CRG.predict(x_val, device)
            result = result[:, window - step:window]
            y_val = torch.as_tensor(y_val).squeeze()[:, window - step:window].cuda()
            mse = loss_function(y_val, result)
            mae = utils.MAE(y_val, result)
            mape = utils.MAPE(y_val, result)
            val_mse_list.append(mse.item())
            val_mae_list.append(mae.item())
            val_mape_list.append(mape.item())
            time2 = time.time()
            forward_time_list.append(time2 - time1)
            # 记录后向过程运行时间，每个epoch记录一次
        path = './time_result/train_{}_{}_{}'.format(args.data_set, args.step_list[0][0], args.batch_size)
        utils.saveTime(path, forward_time_list)
        # 记录测试结果
        val_mse = np.mean(np.array(val_mse_list))
        val_mae = np.mean(np.array(val_mae_list))
        val_mape = np.mean(np.array(val_mape_list))
        del val_mse_list[:]
        del val_mae_list[:]
        del val_mape_list[:]
        if val_mse < best_score:
            torch.save(CRG.state_dict(), 'models/best_{}_{}_{}'.format(dataname, item, step))
            print('val:step:{},mse:{},mae:{},mape:{}'.format(step, val_mse, val_mae, val_mape))
            if (best_score - val_mse) < stop_f:
                stop_flag = True
            best_score = val_mse
        CRG.train()
        valid_mse_list.append(val_mse)
        return best_score, stop_flag


def test(CRG_best, testset, device, dataname, item, step, window, loss_function, results_class, settings, draw_tools):
    with torch.no_grad():
        test_mse_list = []
        test_mae_list = []
        test_mape_list = []
        # 存储每一个 batch 测试结果
        draw_result = []
        draw_y = []
        CRG_best.load_state_dict(torch.load('models/best_{}_{}_{}'.format(dataname, item, step)))
        CRG_best.eval()
        for c, da in enumerate(testset):
            x_test, y_test = da
            if 1 == len(x_test):
                continue
            result = CRG_best.predict(x_test, device)
            result = result[:, window - step:window]
            y_test = torch.as_tensor(y_test).squeeze()[:, window - step:window].cuda()
            draw_y.append(y_test)
            draw_result.append(result)
            # 每个batch画一条
            # draw_tools.draw(y_test[0], result[0])
            mse = loss_function(y_test, result)
            mae = utils.MAE(y_test, result)
            mape = utils.MAPE(y_test, result)
            test_mse_list.append(mse.item())
            test_mae_list.append(mae.item())
            test_mape_list.append(mape.item())
        # 记录测试结果
        draw_y = torch.row_stack(draw_y)
        draw_result = torch.row_stack(draw_result)
        draw_tools.draw_mean(draw_y, draw_result)
        test_mse = np.mean(np.array(test_mse_list))
        test_mae = np.mean(np.array(test_mae_list))
        test_mape = np.mean(np.array(test_mape_list))
        results_class.finalreuslts_dic[step]['mse'] = test_mse
        results_class.finalreuslts_dic[step]['mae'] = test_mae
        results_class.finalreuslts_dic[step]['mape'] = test_mape
        test_result = 'test>>>step:{}, mse:{},mae:{},mape:{}'.format(step, test_mse, test_mae, test_mape)
        print(test_result)
        utils.saveResult(settings, test_result)
        del test_mse_list[:]
        del test_mae_list[:]
        del test_mape_list[:]
        CRG_best.train()