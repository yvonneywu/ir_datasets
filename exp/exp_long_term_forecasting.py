from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings('ignore')

import sys

# import argparse
from .config import Config
from .data_loader_full import load_forecasting_data
# parser = argparse.ArgumentParser(description="Time Series Classification")
# parser.add_argument("--config", type=str, default=None, help="Path to config file")
# parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
# parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
# args = parser.parse_args()



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        ### original data_loader
        data_set, data_loader = data_provider(self.args, flag, vali_test)

        ### load the data 
        # Load config
        config = Config()
        # if args.config and os.path.exists(args.config):
        #     config.load(args.config)

        data_obj = load_forecasting_data(config)
        train_loader = data_obj["train_dataloader"]
        val_loader = data_obj["val_dataloader"]
        test_loader = data_obj["test_dataloader"]

        # return data_set, data_loader
        return train_loader, val_loader, test_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion

    def train(self, setting):
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test', vali_test=True)

        train_loader, vali_loader, test_loader = self._get_data(flag='train')
        train_data, vali_data, test_data = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                # batch_x = batch_x.float().to(self.device) # check shape [batch_size, seq_len, feature]
                # batch_y = batch_y.float().to(self.device)# check shape [batch_size, pred_seq_len, feature]
                # print('batch_x:', batch_x.shape)
                # print('batch_y:', batch_y.shape)

                seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
                future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
                # print('checking data shape', seen_data.shape, future_data.shape)

                ### the original shape is [batch_size, seq_len, feature], pad the se1_dim to 512
                batch_size, seq_len_seen, feature_dim = seen_data.shape
                batch_size, seq_len_future, feature_dim = future_data.shape
                seen_data_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_data.device, dtype=seen_data.dtype)
                future_data_padded = torch.zeros(batch_size, 512, feature_dim, device=future_data.device, dtype=future_data.dtype)

                # Copy the existing data
                seen_data_padded[:, :seq_len_seen, :] = seen_data
                future_data_padded[:, :seq_len_future, :] = future_data

                # Do the same for masks
                seen_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_mask.device, dtype=seen_mask.dtype)
                future_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=future_mask.device, dtype=future_mask.dtype)
                future_mask_padded[:, :seq_len_future, :] = future_mask

                # Use these padded tensors
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()
                
                outputs_dict = self.model(batch_x) # [batch_size, seq_len, feature_dim]
                # print('checking outputs_dict shape', outputs_dict['outputs_time'].shape)
                # print('check device',batch_y.device, outputs_dict['outputs_time'].device)
                
                # loss = criterion(outputs_dict, batch_y)
                future_mask_padded = future_mask_padded.to(self.device)
                mse_loss = ((batch_y - outputs_dict['outputs_time'])**2) * future_mask_padded
                mse_loss = ((batch_y - outputs_dict['outputs_time'])**2)
                mse_loss = mse_loss.sum() / future_mask_padded.sum()
                loss = mse_loss

                train_loss.append(mse_loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                loss.backward()
                model_optim.step()
                loss_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}'.format(epoch + 1, train_steps, train_loss))


            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mae_loss = []

        self.model.in_layer.eval()
        self.model.out_layer.eval()
        self.model.time_proj.eval()
        self.model.text_proj.eval()

        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, batch in enumerate(vali_loader):
                # batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float()

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
                future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
                # print('checking data shape', seen_data.shape, future_data.shape)

                ### the original shape is [batch_size, seq_len, feature], pad the se1_dim to 512
                batch_size, seq_len_seen, feature_dim = seen_data.shape
                batch_size, seq_len_future, feature_dim = future_data.shape
                seen_data_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_data.device, dtype=seen_data.dtype)
                future_data_padded = torch.zeros(batch_size, 512, feature_dim, device=future_data.device, dtype=future_data.dtype)

                # Copy the existing data
                seen_data_padded[:, :seq_len_seen, :] = seen_data
                future_data_padded[:, :seq_len_future, :] = future_data

                # Do the same for masks
                seen_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_mask.device, dtype=seen_mask.dtype)
                future_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=future_mask.device, dtype=future_mask.dtype)
                future_mask_padded[:, :seq_len_future, :] = future_mask

                # Use these padded tensors
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()

                outputs = self.model(batch_x)

                # outputs_ensemble = outputs['outputs_time']
                # # encoder - decoder
                # outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                # pred = outputs_ensemble.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = F.mse_loss(pred, true)
                # loss = criterion(outputs_dict, batch_y)
                future_mask_padded = future_mask_padded.to(self.device)
                mse_loss = ((batch_y - outputs['outputs_time'])**2) * future_mask_padded
                mse_loss = mse_loss.sum() / future_mask_padded.sum()
                # mse_loss = F.mse_loss(outputs['outputs_time'], batch_y, reduction='none')
                loss = mse_loss

                if loss.item() < 1000 and not torch.isnan(loss) and not torch.isinf(loss):
                    # Skip extreme values
                    total_loss.append(loss.item())
                    # Skip extreme values
                    # if mse_loss.item() < 1000 and not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                    #     total_mse += mse_loss.item()
                    #     total_mae += mae_loss.item()
                    #     batch_count += 1

                # mae_loss = (batch_y - outputs['outputs_time']).abs()
                # mae_loss = mae_loss * future_mask_padded
                # mae_loss = mae_loss.sum() / future_mask_padded.sum()

                # total_loss.append(loss.item())
                # total_mae_loss.append(mae_loss.item())

        total_loss = np.average(total_loss)
        # total_mae = np.average(total_mae_loss)

        self.model.in_layer.train()
        self.model.out_layer.train()
        self.model.time_proj.train()
        self.model.text_proj.train()

        return total_loss

    def test(self, setting, test=0):
        # zero shot
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        # test_data, test_loader = self._get_data(flag='test')
        train_loader, vali_loader, test_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        batch_count = 0
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, batch in enumerate(test_loader):
                # batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float().to(self.device)

                seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
                future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
                # print('checking data shape', seen_data.shape, future_data.shape)

                ### the original shape is [batch_size, seq_len, feature], pad the se1_dim to 512
                batch_size, seq_len_seen, feature_dim = seen_data.shape
                batch_size, seq_len_future, feature_dim = future_data.shape
                seen_data_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_data.device, dtype=seen_data.dtype)
                future_data_padded = torch.zeros(batch_size, 512, feature_dim, device=future_data.device, dtype=future_data.dtype)

                # Copy the existing data
                seen_data_padded[:, :seq_len_seen, :] = seen_data
                future_data_padded[:, :seq_len_future, :] = future_data

                # Do the same for masks
                seen_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_mask.device, dtype=seen_mask.dtype)
                future_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=future_mask.device, dtype=future_mask.dtype)
                future_mask_padded[:, :seq_len_future, :] = future_mask
                future_mask_padded = future_mask_padded.to(self.device)

                # Use these padded tensors
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()

                # outputs = self.model(batch_x[:, -self.args.seq_len:, :])

                # outputs_ensemble = outputs['outputs_time']

                outputs = self.model(batch_x)
                
                # outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                # batch_y = batch_y[:, -self.args.pred_len:, :]

                # pred = outputs_ensemble.detach().cpu().numpy() 
                # true = batch_y.detach().cpu().numpy() 

                # preds.append(pred)
                # trues.append(true)

                future_mask_padded = future_mask_padded.to(self.device)
                mse_loss = ((batch_y - outputs['outputs_time'])**2) * future_mask_padded
                mse_loss = mse_loss.sum() / future_mask_padded.sum()
                # mse_loss = F.mse_loss(outputs['outputs_time'], batch_y, reduction='none')
                loss = mse_loss

                if loss.item() < 1000 and not torch.isnan(loss) and not torch.isinf(loss):
                    # Skip extreme values
                    
                    mae_loss = (batch_y - outputs['outputs_time']).abs()
                    mae_loss = mae_loss * future_mask_padded
                    mae_loss = mae_loss.sum() / future_mask_padded.sum()

                    total_mse += mse_loss.item()
                    total_mae += mae_loss.item()
                    batch_count += 1

        fina_mse = total_mse / batch_count
        fina_mae = total_mae / batch_count
        print('final mse', fina_mse)
        print('final mae', fina_mae)
        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
