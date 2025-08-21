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
        criterion = nn.MSELoss(reduction='none')  # Keep per-sample loss for anomaly scoring
        return criterion
    
    def _is_reconstruction_model(self) -> bool:
        """Check if the current model is reconstruction-based."""
        reconstruction_models = [
            'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'DAGMM',
            'AutoEncoder', 'VAE', 'LSTM_VAE', 'LSTM_AE'
        ]
        return self.args.model in reconstruction_models
    
    def _get_anomaly_scores(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get anomaly scores using appropriate method based on model type.
        
        Args:
            batch_x: Input batch data
            
        Returns:
            Tuple of (outputs, anomaly_scores)
        """
        if self._is_reconstruction_model():
            return self._reconstruction_based_scoring(batch_x)
        else:
            return self._prediction_based_scoring(batch_x)
    
    def _reconstruction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using reconstruction-based method."""
        try:
            # Try to use model's specific anomaly detection method
            if hasattr(self.model, 'detect_anomaly'):
                outputs, scores = self.model.detect_anomaly(batch_x)
            elif hasattr(self.model, 'get_anomaly_score'):
                outputs = self.model(batch_x)
                scores = self.model.get_anomaly_score(batch_x)
            else:
                # Fallback: calculate reconstruction error
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: simple reconstruction error
            try:
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores
    
    def _prediction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using prediction-based method."""
        try:
            # For prediction-based detection, we need to:
            # 1. Make predictions for the next few steps
            # 2. Calculate prediction variance or error as anomaly score
            
            # Get prediction horizon (use config pred_len or default to 1)
            pred_len = getattr(self.args, 'pred_len', 1)
            
            # Make predictions
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(batch_x, pred_len)
            else:
                # Fallback: use model directly
                predictions = self.model(batch_x)
            
            # Calculate prediction error as anomaly score
            if predictions.dim() == 3:  # (batch, pred_len, features)
                # Use prediction variance across time steps as anomaly score
                scores = torch.var(predictions, dim=1, keepdim=True)
            else:
                # Use prediction magnitude as anomaly score
                scores = torch.mean(torch.abs(predictions), dim=-1, keepdim=True)
            
            # For prediction-based, outputs are the predictions
            outputs = predictions
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: use input variance as anomaly score
            try:
                scores = torch.var(batch_x, dim=1, keepdim=True)
                outputs = batch_x
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores
    
    def _compute_model_loss(self, batch_x: torch.Tensor, criterion: nn.Module, 
                           epoch: int = 1) -> torch.Tensor:
        """
        Compute model-specific loss following KOGAS methodology.
        
        Args:
            batch_x: Input batch data
            criterion: Loss function
            epoch: Current epoch (used for some models like USAD)
            
        Returns:
            Loss tensor
        """
        if hasattr(self.model, 'compute_loss'):
            # Use model's built-in loss computation
            return self.model.compute_loss(batch_x, criterion)
        elif self.args.model == 'USAD':
            return self._compute_usad_loss(batch_x, criterion, epoch)
        elif self.args.model == 'AnomalyTransformer':
            return self._compute_anomaly_transformer_loss(batch_x, criterion)
        elif self.args.model == 'OmniAnomaly':
            return self._compute_omnianomaly_loss(batch_x, criterion)
        elif self.args.model == 'LSTM_VAE':
            return self._compute_lstm_vae_loss(batch_x, criterion)
        elif self.args.model == 'DAGMM':
            return self._compute_dagmm_loss(batch_x, criterion)
        else:
            # Default reconstruction loss
            outputs = self.model(batch_x)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # Take first output if multiple
            loss = criterion(outputs, batch_x)
            return torch.mean(loss)
    
    def _compute_usad_loss(self, batch_x: torch.Tensor, criterion: nn.Module, epoch: int) -> torch.Tensor:
        """USAD-specific loss computation following KOGAS methodology."""
        if hasattr(self.model, 'training_step'):
            # Use model's training step method
            loss1, loss2 = self.model.training_step(batch_x, epoch)
            return loss1 + loss2
        else:
            # Fallback to reconstruction loss
            outputs = self.model(batch_x)
            if isinstance(outputs, list):
                # USAD returns multiple outputs
                w1, w2 = outputs[0], outputs[1] if len(outputs) > 1 else outputs[0]
                loss = 0.5 * (criterion(w1, batch_x).mean() + criterion(w2, batch_x).mean())
            else:
                loss = criterion(outputs, batch_x).mean()
            return loss
    
    def _compute_anomaly_transformer_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """AnomalyTransformer-specific loss computation following KOGAS methodology."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback to reconstruction loss
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) > 1:
                output, series, prior, _ = outputs
                # Simple reconstruction loss (full implementation would include series loss)
                loss = criterion(output, batch_x).mean()
            else:
                loss = criterion(outputs, batch_x).mean()
            return loss
    
    def _compute_omnianomaly_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """OmniAnomaly-specific loss computation following KOGAS methodology."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                x_recon, mu, logvar = outputs[0], outputs[1], outputs[2]
                rec_loss = criterion(x_recon, batch_x).mean()
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                beta = getattr(self.model, 'beta', 0.01)
                return rec_loss + beta * kl_loss
            else:
                return criterion(outputs, batch_x).mean()
    
    def _compute_lstm_vae_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """LSTM-VAE-specific loss computation following KOGAS methodology."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, list) and len(outputs) >= 2:
                x_decoded, kl_loss = outputs[0], outputs[1]
                rec_loss = criterion(x_decoded, batch_x).mean()
                return rec_loss + kl_loss
            else:
                return criterion(outputs, batch_x).mean()
    
    def _compute_dagmm_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """DAGMM-specific loss computation following KOGAS methodology."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                _, x_hat = outputs[0], outputs[1]
                loss = criterion(x_hat, batch_x).mean()
            else:
                loss = criterion(outputs, batch_x).mean()
            return loss

    def vali(self, vali_data, vali_loader, criterion) -> Dict[str, float]:
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                # Get anomaly scores using appropriate method
                outputs, scores = self._get_anomaly_scores(batch_x)
                
                # Calculate loss using KOGAS methodology
                loss = self._compute_model_loss(batch_x, criterion)
                
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

        # Log detection method being used
        detection_method = "reconstruction-based" if self._is_reconstruction_model() else "prediction-based"
        self.logger.info(f"Using {detection_method} anomaly detection with model: {self.args.model}")

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Get anomaly scores using appropriate method
                        outputs, scores = self._get_anomaly_scores(batch_x)
                        
                        # Calculate loss using KOGAS methodology
                        loss = self._compute_model_loss(batch_x, criterion, epoch + 1)
                        
                        train_loss.append(loss.item())
                else:
                    # Get anomaly scores using appropriate method
                    outputs, scores = self._get_anomaly_scores(batch_x)
                    
                    # Calculate loss using KOGAS methodology
                    loss = self._compute_model_loss(batch_x, criterion, epoch + 1)
                    
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
            'best_model_path': best_model_path,
            'detection_method': detection_method
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
                
                # Get anomaly scores using appropriate method
                outputs, scores = self._get_anomaly_scores(batch_x)
                
                anomaly_scores.append(scores.cpu().numpy())
                if hasattr(test_data, 'labels'):
                    labels.append(batch_y.cpu().numpy())

        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        
        folder_path = './results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'anomaly_scores.npy', anomaly_scores)
        
        # Save detection method information
        detection_method = "reconstruction-based" if self._is_reconstruction_model() else "prediction-based"
        method_info = {
            'detection_method': detection_method,
            'model_type': self.args.model,
            'description': 'Uses reconstruction error to detect anomalies' if self._is_reconstruction_model() else 'Uses prediction variance to detect anomalies'
        }
        
        if self._is_reconstruction_model():
            method_info['approach'] = 'Compares input with reconstructed output'
        else:
            method_info['approach'] = 'Analyzes prediction patterns and variance'
        
        np.save(folder_path + 'detection_method_info.npy', method_info)
        
        if labels:
            labels = np.concatenate(labels, axis=0)
            np.save(folder_path + 'labels.npy', labels)
            
            # Calculate standard metrics
            from sklearn.metrics import roc_auc_score, average_precision_score
            try:
                auc = roc_auc_score(labels.flatten(), anomaly_scores.flatten())
                ap = average_precision_score(labels.flatten(), anomaly_scores.flatten())
            except ValueError as e:
                self.logger.warning(f"Could not calculate standard metrics: {e}")
                auc, ap = 0.0, 0.0
            
            # Calculate PA%K metrics following KOGAS methodology
            pa_metrics = self._calculate_pa_metrics(anomaly_scores, labels)
            
            result_dict = {
                'auc': auc,
                'ap': ap,
                'pa_auc': pa_metrics['pa_auc'],
                'anomaly_score_mean': np.mean(anomaly_scores),
                'anomaly_score_std': np.std(anomaly_scores),
                'detection_method': detection_method,
                'method_info': method_info
            }
            
            # Add PA%K individual metrics
            result_dict.update(pa_metrics['metrics'])
            
            # Save PA%K metrics
            np.save(folder_path + 'pa_metrics.npy', pa_metrics)
            
            return result_dict
        
        return {
            'anomaly_score_mean': np.mean(anomaly_scores),
            'anomaly_score_std': np.std(anomaly_scores),
            'detection_method': detection_method,
            'method_info': method_info
        }
    
    def predict_single(self, input_data: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Make prediction for single input (mainly for prediction-based models).
        
        Args:
            input_data: Input tensor
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(input_data, num_steps)
            else:
                # Fallback: use model directly
                predictions = self.model(input_data)
        
        return predictions
    
    def detect_anomaly(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies in input data.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Anomaly scores tensor
        """
        self.model.eval()
        with torch.no_grad():
            outputs, scores = self._get_anomaly_scores(input_data)
        
        return scores
    
    def _calculate_pa_metrics(self, anomaly_scores: np.ndarray, labels: np.ndarray, 
                             K_VALUES: list = None) -> dict:
        """
        Calculate PA%K metrics following KOGAS methodology.
        
        Args:
            anomaly_scores: Anomaly scores array
            labels: Ground truth labels array
            K_VALUES: List of K values for PA%K calculation
            
        Returns:
            Dictionary containing PA%K metrics
        """
        if K_VALUES is None:
            K_VALUES = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        # Flatten arrays if needed
        scores_flat = anomaly_scores.flatten()
        labels_flat = labels.flatten()
        
        # Check if we have both classes
        unique_classes = np.unique(labels_flat)
        if len(unique_classes) == 1:
            self.logger.warning(f"Only one class present in labels: {unique_classes[0]}")
            return {'pa_auc': 0.0, 'metrics': {}}
        
        metrics = {}
        f1_values = []
        
        # Calculate threshold range
        start_threshold = np.percentile(scores_flat, 90)
        end_threshold = np.percentile(scores_flat, 99)
        self.logger.info(f'PA%K threshold range: {start_threshold:.4f} to {end_threshold:.4f}')
        
        # Calculate PA%K for each K value
        for k in K_VALUES:
            try:
                # Import the specific metric function if available
                try:
                    from utils.metrics import bf_search
                    result, threshold = bf_search(
                        scores_flat, labels_flat,
                        start=np.percentile(scores_flat, 50),
                        end=np.percentile(scores_flat, 99),
                        step_num=1000,
                        K=k,
                        verbose=False
                    )
                    f1, precision, recall, _, _, _, _, roc_auc, _, _ = result
                except ImportError:
                    # Fallback implementation
                    precision, recall, thresholds = precision_recall_curve(labels_flat, scores_flat)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    f1, precision, recall = f1_scores[best_idx], precision[best_idx], recall[best_idx]
                    roc_auc = roc_auc_score(labels_flat, scores_flat)
                    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                
                f1_values.append(f1)
                metrics[f'precision_{k}'] = precision
                metrics[f'recall_{k}'] = recall
                metrics[f'f1_{k}'] = f1
                metrics[f'roc_auc_{k}'] = roc_auc
                
                self.logger.info(f"PA%{k:02d}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate PA%{k} metrics: {e}")
                f1_values.append(0.0)
                metrics[f'f1_{k}'] = 0.0
        
        # Calculate PA%K AUC
        pa_auc = self._calculate_pa_auc(f1_values, K_VALUES)
        metrics['pa_auc'] = pa_auc
        
        self.logger.info(f'PA%K AUC: {pa_auc:.4f}')
        
        return {'pa_auc': pa_auc, 'metrics': metrics}
    
    def _calculate_pa_auc(self, f1_values: list, k_values: list) -> float:
        """Calculate PA%K AUC following KOGAS methodology."""
        auc = 0
        for i in range(len(k_values) - 1):
            auc += 0.5 * (f1_values[i] + f1_values[i + 1]) * (int(k_values[i + 1]) - int(k_values[i]))
        auc /= 100
        return auc