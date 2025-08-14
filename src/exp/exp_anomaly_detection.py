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
from core.base_model import BaseAnomalyDetectionModel
import warnings
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        
    def _build_model(self) -> BaseAnomalyDetectionModel:
        model = self.model_dict.get(self.args.model, self.model_dict['AnomalyTransformer']).Model(self.args).float()
        
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

    def vali(self, vali_data, vali_loader, criterion) -> Dict[str, float]:
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                if hasattr(self.model, 'detect_anomaly'):
                    outputs, scores = self.model.detect_anomaly(batch_x)
                else:
                    outputs = self.model(batch_x)
                    scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
                
                loss = criterion(outputs, batch_x)
                total_loss.append(loss.item())
                
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

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        train_history = []
        val_history = []

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if hasattr(self.model, 'detect_anomaly'):
                            outputs, scores = self.model.detect_anomaly(batch_x)
                        else:
                            outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_x)
                        train_loss.append(loss.item())
                else:
                    if hasattr(self.model, 'detect_anomaly'):
                        outputs, scores = self.model.detect_anomaly(batch_x)
                    else:
                        outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_x)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            train_metrics = {'loss': train_loss}
            val_metrics = vali_loss
            
            train_history.append(train_metrics)
            val_history.append(val_metrics)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss['loss']:.7f}")
            
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

        anomaly_scores = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                if hasattr(self.model, 'get_anomaly_score'):
                    scores = self.model.get_anomaly_score(batch_x)
                else:
                    outputs = self.model(batch_x)
                    scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
                
                anomaly_scores.append(scores.cpu().numpy())
                if hasattr(test_data, 'labels'):
                    labels.append(batch_y.cpu().numpy())

        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        
        folder_path = './results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'anomaly_scores.npy', anomaly_scores)
        
        if labels:
            labels = np.concatenate(labels, axis=0)
            np.save(folder_path + 'labels.npy', labels)
            
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(labels.flatten(), anomaly_scores.flatten())
            ap = average_precision_score(labels.flatten(), anomaly_scores.flatten())
            
            return {
                'auc': auc,
                'ap': ap,
                'anomaly_score_mean': np.mean(anomaly_scores),
                'anomaly_score_std': np.std(anomaly_scores)
            }
        
        return {
            'anomaly_score_mean': np.mean(anomaly_scores),
            'anomaly_score_std': np.std(anomaly_scores)
        }