"""
Base Trainer Module

This module provides the base training functionality for time series models,
including training loops, validation, testing, and checkpoint management.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[int] = None


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    val_loss: float
    val_metric: float
    epoch: int


class BaseTrainer(ABC):
    """Abstract base class for training time series models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: Neural network model to train
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Initialize components
        self.optimizer = optimizer or self._build_optimizer()
        self.scheduler = scheduler or self._build_scheduler()
        self.criterion = criterion or self._build_criterion()
        self.device = device or self._acquire_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging and directories
        self._setup_logging()
        self._setup_checkpoint_dir()
        
        # Training state
        self.start_epoch = 0
        self.best_metric = float('inf')
        self.early_stop_counter = 0
        
        # Mixed precision training
        self.use_amp = getattr(config, 'use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logging.info(f"Base trainer initialized on device: {self.device}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _setup_checkpoint_dir(self) -> None:
        """Setup checkpoint directory."""
        self.checkpoint_dir = Path(self.config.checkpoints) / self.config.des
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _acquire_device(self) -> torch.device:
        """Acquire and configure the training device."""
        if self.config.use_gpu and self.config.gpu_type == 'cuda':
            if not self.config.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
                device = torch.device(f'cuda:{self.config.gpu}')
            else:
                device = torch.device('cuda')
                self.model = nn.DataParallel(
                    self.model, 
                    device_ids=[int(x) for x in self.config.devices.split(',')]
                )
        elif self.config.use_gpu and self.config.gpu_type == 'mps':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _build_optimizer(self) -> Optimizer:
        """Build the optimizer for training."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def _build_scheduler(self) -> Optional[_LRScheduler]:
        """Build the learning rate scheduler."""
        if self.config.lradj == 'type1':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=0.5
            )
        elif self.config.lradj == 'type2':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train_epochs
            )
        return None
    
    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        if self.config.loss == 'MSE':
            return nn.MSELoss()
        elif self.config.loss == 'MAE':
            return nn.L1Loss()
        elif self.config.loss == 'Huber':
            return nn.HuberLoss()
        else:
            self.logger.warning(f"Unknown loss function: {self.config.loss}, using MSE")
            return nn.MSELoss()
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch. Must be implemented by subclasses."""
        raise NotImplementedError
    
    @abstractmethod
    def validate_epoch(self, epoch: int) -> ValidationMetrics:
        """Validate for one epoch. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training results and metrics
        """
        logging.info("Starting training...")
        
        best_val_metric = float('inf')
        training_history = []
        
        for epoch in range(self.start_epoch, self.config.train_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                train_metrics.val_loss = val_metrics.val_loss
                
                # Check if this is the best model
                if val_metrics.val_metric < best_val_metric:
                    best_val_metric = val_metrics.val_metric
                    self.best_metric = best_val_metric
                    self._save_checkpoint(epoch, is_best=True)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                # Early stopping
                if self.early_stop_counter >= self.config.patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                # No validation, save checkpoint periodically
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch, is_best=False)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                train_metrics.learning_rate = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_metrics(epoch, train_metrics, epoch_time)
            
            # Store history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics.train_loss,
                'val_loss': train_metrics.val_loss,
                'learning_rate': train_metrics.learning_rate,
                'epoch_time': epoch_time
            })
        
        # Save final checkpoint
        self._save_checkpoint(self.config.train_epochs - 1, is_best=False)
        
        logging.info("Training completed")
        return {
            'training_history': training_history,
            'best_val_metric': best_val_metric,
            'final_epoch': epoch
        }
    
    def _log_epoch_metrics(self, epoch: int, metrics: TrainingMetrics, epoch_time: float) -> None:
        """Log metrics for the current epoch."""
        log_msg = f"Epoch {epoch + 1}/{self.config.train_epochs}"
        log_msg += f" - Train Loss: {metrics.train_loss:.6f}"
        
        if metrics.val_loss is not None:
            log_msg += f" - Val Loss: {metrics.val_loss:.6f}"
        
        if metrics.learning_rate is not None:
            log_msg += f" - LR: {metrics.learning_rate:.6f}"
        
        log_msg += f" - Time: {epoch_time:.2f}s"
        
        self.logger.info(log_msg)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"Best model saved at epoch {epoch + 1}")
        
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch']
    
    def test(self) -> Dict[str, Any]:
        """
        Test the model on test data.
        
        Returns:
            Dictionary containing test results
        """
        if self.test_loader is None:
            raise ValueError("Test loader not provided")
        
        self.logger.info("Starting model testing...")
        
        self.model.eval()
        test_losses = []
        test_metrics = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                test_losses.append(loss.item())
                
                # Calculate additional metrics
                batch_metrics = self._calculate_batch_metrics(output, target)
                test_metrics.append(batch_metrics)
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Test batch {batch_idx}: loss={loss.item():.6f}")
        
        # Aggregate results
        avg_test_loss = np.mean(test_losses)
        aggregated_metrics = self._aggregate_metrics(test_metrics)
        
        results = {
            'test_loss': avg_test_loss,
            'metrics': aggregated_metrics
        }
        
        self.logger.info(f"Testing completed. Average loss: {avg_test_loss:.6f}")
        return results
    
    def _calculate_batch_metrics(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate metrics for a single batch.
        
        Args:
            output: Model output
            target: Target values
            
        Returns:
            Dictionary of batch metrics
        """
        # Default implementation - can be overridden by subclasses
        with torch.no_grad():
            mse = nn.functional.mse_loss(output, target).item()
            mae = nn.functional.l1_loss(output, target).item()
            
            return {
                'mse': mse,
                'mae': mae
            }
    
    def _aggregate_metrics(self, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across batches.
        
        Args:
            batch_metrics: List of batch metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not batch_metrics:
            return {}
        
        aggregated = {}
        for key in batch_metrics[0].keys():
            values = [metrics[key] for metrics in batch_metrics]
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def predict(self, data_loader: Optional[DataLoader] = None) -> torch.Tensor:
        """
        Generate predictions using the trained model.
        
        Args:
            data_loader: Data loader for prediction (uses test_loader if None)
            
        Returns:
            Model predictions
        """
        if data_loader is None:
            data_loader = self.test_loader
        
        if data_loader is None:
            raise ValueError("No data loader provided for prediction")
        
        self.logger.info("Starting prediction...")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="Generating predictions"):
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(predictions, dim=0)
        
        self.logger.info(f"Prediction completed. Shape: {all_predictions.shape}")
        return all_predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'optimizer': self.optimizer.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__ if self.scheduler else None,
            'criterion': self.criterion.__class__.__name__,
            'use_amp': self.use_amp
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'scaler'):
            del self.scaler
        
        torch.cuda.empty_cache()
        self.logger.info("Trainer cleanup completed")


class ForecastingTrainer(BaseTrainer):
    """Specialized trainer for forecasting tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Forecasting trainer initialized")
    
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch (forecasting-specific implementation)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}: loss={loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return TrainingMetrics(train_loss=avg_loss, epoch=epoch)
    
    def validate_epoch(self, epoch: int) -> ValidationMetrics:
        """Validate for one epoch (forecasting-specific implementation)."""
        if self.val_loader is None:
            raise ValueError("Validation loader not provided")
        
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate additional validation metric (e.g., MSE)
                metric = nn.functional.mse_loss(output, target)
                
                total_loss += loss.item()
                total_metric += metric.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return ValidationMetrics(
            val_loss=avg_loss,
            val_metric=avg_metric,
            epoch=epoch
        )