import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from core.base_trainer import ForecastingTrainer
from core.base_model import BaseForecastingModel
import warnings
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self) -> BaseForecastingModel:
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def detect_anomaly(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies using prediction-based method.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Anomaly scores tensor
        """
        self.model.eval()
        with torch.no_grad():
            # For prediction-based anomaly detection, we:
            # 1. Make predictions for the next few steps
            # 2. Calculate prediction variance or error as anomaly score
            
            # Get prediction horizon
            pred_len = getattr(self.args, 'pred_len', 1)
            
            # Prepare input for forecasting
            batch_size, seq_len, features = input_data.shape
            
            # Create dummy decoder input (required for some models)
            if hasattr(self.args, 'label_len'):
                label_len = self.args.label_len
            else:
                label_len = seq_len // 2
            
            # Create dummy decoder input
            dec_inp = torch.zeros(batch_size, pred_len, features, device=input_data.device)
            
            # Make predictions
            if 'Linear' in self.args.model or 'TST' in self.args.model:
                predictions = self.model(input_data)
            else:
                # For transformer-based models, we need decoder input
                # Since we're doing anomaly detection, we'll use the input as decoder input
                if hasattr(self.args, 'output_attention') and self.args.output_attention:
                    predictions = self.model(input_data, input_data, dec_inp, dec_inp)[0]
                else:
                    predictions = self.model(input_data, input_data, dec_inp, dec_inp)
            
            # Calculate anomaly scores based on prediction patterns
            if predictions.dim() == 3:  # (batch, pred_len, features)
                # Use prediction variance across time steps as anomaly score
                anomaly_scores = torch.var(predictions, dim=1, keepdim=True)
            else:
                # Use prediction magnitude as anomaly score
                anomaly_scores = torch.mean(torch.abs(predictions), dim=-1, keepdim=True)
            
            return anomaly_scores
    
    def predict_single(self, input_data: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Make prediction for single input.
        
        Args:
            input_data: Input tensor
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            batch_size, seq_len, features = input_data.shape
            
            # Create dummy decoder input
            dec_inp = torch.zeros(batch_size, num_steps, features, device=input_data.device)
            
            # Make predictions
            if 'Linear' in self.args.model or 'TST' in self.args.model:
                predictions = self.model(input_data)
            else:
                # For transformer-based models
                if hasattr(self.args, 'output_attention') and self.args.output_attention:
                    predictions = self.model(input_data, input_data, dec_inp, dec_inp)[0]
                else:
                    predictions = self.model(input_data, input_data, dec_inp, dec_inp)
            
            # Return predictions for the requested number of steps
            if predictions.shape[1] >= num_steps:
                return predictions[:, :num_steps, :]
            else:
                return predictions

    def vali(self, vali_data, vali_loader, criterion) -> Dict[str, float]:
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return {'loss': total_loss}

    def train(self) -> Dict[str, Any]:
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.des)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        train_history = []
        val_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.logger.info(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            train_metrics = {'loss': train_loss}
            val_metrics = vali_loss
            test_metrics = test_loss
            
            train_history.append(train_metrics)
            val_history.append(val_metrics)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss['loss']:.7f} Test Loss: {test_loss['loss']:.7f}")
            
            early_stopping(vali_loss['loss'], self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_model_path': best_model_path
        }

    def test(self) -> Dict[str, float]:
        test_data, test_loader = self._get_data(flag='test')
        
        if hasattr(self.args, 'test') and self.args.test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.des, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        folder_path = './results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # Fallback metrics not provided by utils.metrics
        from utils.metrics import RSE, CORR
        rse = RSE(preds, trues)
        corr = CORR(preds, trues)
        self.logger.info(f'mse:{mse}, mae:{mae}, rse:{rse}, corr:{corr}')
        f = open("result.txt", 'a')
        f.write(self.args.des + "  \n")
        f.write(f'mse:{mse}, mae:{mae}, rse:{rse}, corr:{corr}')
        f.write('\n')
        f.write('\n')
        f.close()

        # Save metrics as a dict to support vector-valued corr
        metrics_dict = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'rse': float(rse),
            'corr': corr
        }
        np.save(folder_path + 'metrics.npy', metrics_dict, allow_pickle=True)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'rse': rse,
            'corr': corr
        }