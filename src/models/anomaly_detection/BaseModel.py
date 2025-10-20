"""
Base Model for Anomaly Detection

This module provides the base class for all anomaly detection models,
defining common interfaces and methods that all models should implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np


class BaseAnomalyDetectionModel(nn.Module, ABC):
    """
    Base class for all anomaly detection models.
    
    This class defines the common interface that all anomaly detection models
    should implement, ensuring consistency across different model architectures.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the base anomaly detection model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super(BaseAnomalyDetectionModel, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, feature_size)
            
        Returns:
            Model output tensor
        """
        pass
    
    @abstractmethod
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input data (for reconstruction-based models).
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, feature_size)
            
        Returns:
            Reconstructed tensor with same shape as input
        """
        pass
    
    def detect_anomaly(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in input data.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, feature_size)
            
        Returns:
            Tuple of (reconstructed_output, anomaly_scores)
        """
        reconstructed = self.reconstruct(x)
        anomaly_scores = self._calculate_anomaly_scores(x, reconstructed)
        return reconstructed, anomaly_scores
    
    def _calculate_anomaly_scores(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly scores based on reconstruction error.
        
        Args:
            x: Original input tensor
            reconstructed: Reconstructed tensor
            
        Returns:
            Anomaly scores tensor with shape (batch_size, seq_len)
        """
        # Default: MSE-based anomaly score
        return torch.mean((x - reconstructed) ** 2, dim=-1)
    
    def get_anomaly_score(self, x: torch.Tensor, method: str = 'mse') -> torch.Tensor:
        """
        Calculate anomaly scores using different methods.
        
        Args:
            x: Input tensor
            method: Scoring method ('mse', 'mae', 'combined')
            
        Returns:
            Anomaly scores tensor
        """
        reconstructed = self.reconstruct(x)
        
        if method == 'mse':
            return torch.mean((x - reconstructed) ** 2, dim=-1)
        elif method == 'mae':
            return torch.mean(torch.abs(x - reconstructed), dim=-1)
        elif method == 'combined':
            mse = torch.mean((x - reconstructed) ** 2, dim=-1)
            mae = torch.mean(torch.abs(x - reconstructed), dim=-1)
            return 0.5 * (mse + mae)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def compute_loss(self, x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            x: Input tensor
            criterion: Loss function
            
        Returns:
            Loss tensor
        """
        reconstructed = self.reconstruct(x)
        loss = criterion(reconstructed, x)
        return torch.mean(loss)
    
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                   epoch: int = 1, **kwargs) -> Tuple[float, np.ndarray]:
        """
        Perform a single training step.
        
        Args:
            batch_x: Input tensor
            batch_y: Target tensor (usually same as input for reconstruction)
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            **kwargs: Additional arguments for model-specific training
            
        Returns:
            Tuple of (loss_value, anomaly_scores)
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self.forward(batch_x)
        
        # Compute loss
        loss = criterion(output, batch_y)
        
        # Check for NaN or Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf loss detected at epoch {epoch}")
            return float('inf'), np.zeros((batch_x.shape[0], batch_x.shape[1]))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional)
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Calculate anomaly scores
        with torch.no_grad():
            reconstructed = self.reconstruct(batch_x)
            anomaly_scores = self._calculate_anomaly_scores(batch_x, reconstructed)
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
        
        return loss.item(), anomaly_scores
    
    def validation_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                       criterion: nn.Module, **kwargs) -> Tuple[float, np.ndarray]:
        """
        Perform a single validation step.
        
        Args:
            batch_x: Input tensor
            batch_y: Target tensor
            criterion: Loss function
            **kwargs: Additional arguments for model-specific validation
            
        Returns:
            Tuple of (loss_value, anomaly_scores)
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            output = self.forward(batch_x)
            
            # Compute loss
            loss = criterion(output, batch_y)
            
            # Calculate anomaly scores
            reconstructed = self.reconstruct(batch_x)
            anomaly_scores = self._calculate_anomaly_scores(batch_x, reconstructed)
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
        
        return loss.item(), anomaly_scores
    
    def test_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                  criterion: nn.Module, **kwargs) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Perform a single test step.
        
        Args:
            batch_x: Input tensor
            batch_y: Target tensor
            criterion: Loss function
            **kwargs: Additional arguments for model-specific testing
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            predictions = self.forward(batch_x)
            
            # Calculate anomaly scores
            reconstructed = self.reconstruct(batch_x)
            anomaly_scores = self._calculate_anomaly_scores(batch_x, reconstructed)
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
        
        return predictions, anomaly_scores
    
    def train_epoch(self, train_loader, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module, epoch: int = 1, **kwargs) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with training metrics
        """
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_x, batch_y, *_) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Perform training step
            loss, _ = self.train_step(batch_x, batch_y, optimizer, criterion, epoch, **kwargs)
            
            total_loss += loss
            num_batches += 1
            
            # Log progress if needed
            if hasattr(self.config, 'log_interval') and batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def validate_epoch(self, val_loader, criterion: nn.Module, **kwargs) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with validation metrics
        """
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Perform validation step
                loss, _ = self.validation_step(batch_x, batch_y, criterion, **kwargs)
                
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def test_epoch(self, test_loader, criterion: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Test for one epoch.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with test metrics and predictions
        """
        self.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_anomaly_scores = []
        
        with torch.no_grad():
            for batch_x, batch_y, *_ in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Perform test step
                predictions, anomaly_scores = self.test_step(batch_x, batch_y, criterion, **kwargs)
                
                # Calculate loss for monitoring
                loss = criterion(predictions, batch_y).mean().item()
                total_loss += loss
                num_batches += 1
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_anomaly_scores.append(anomaly_scores)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'test_loss': avg_loss,
            'num_batches': num_batches,
            'predictions': torch.cat(all_predictions, dim=0) if all_predictions else None,
            'anomaly_scores': np.concatenate(all_anomaly_scores, axis=0) if all_anomaly_scores else None
        }
    
    def fit(self, train_loader, val_loader, optimizer: torch.optim.Optimizer, 
            criterion: nn.Module, epochs: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            epochs: Number of epochs to train
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with training history
        """
        train_history = []
        val_history = []
        
        for epoch in range(1, epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch, **kwargs)
            train_history.append(train_metrics)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, criterion, **kwargs)
                val_history.append(val_metrics)
                
                print(f'Epoch {epoch}/{epochs} - Train Loss: {train_metrics["train_loss"]:.6f}, '
                      f'Val Loss: {val_metrics["val_loss"]:.6f}')
            else:
                print(f'Epoch {epoch}/{epochs} - Train Loss: {train_metrics["train_loss"]:.6f}')
        
        return {
            'train_history': train_history,
            'val_history': val_history
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
        self.eval()
        with torch.no_grad():
            return self.forward(input_data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter count.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params if total_params > 0 else 0,
            'device': str(self.device)
        }
    
    def save_checkpoint(self, filepath: str, epoch: int = None, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            **kwargs: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'epoch': epoch,
            **kwargs
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, map_location: str = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            map_location: Device to map tensors to
            
        Returns:
            Checkpoint dictionary
        """
        if map_location is None:
            map_location = str(self.device)
        
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint
    
    def freeze_parameters(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_summary(self) -> Dict[str, int]:
        """
        Get summary of model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total = self.count_parameters()
        trainable = self.count_trainable_parameters()
        
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'frozen_parameters': total - trainable,
            'trainable_ratio': trainable / total if total > 0 else 0
        }
    
    def to_device(self, device: Union[str, torch.device]) -> 'BaseAnomalyDetectionModel':
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        return self.to(self.device)
    
    def __str__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return f"{info['model_name']}(parameters={info['total_parameters']}, trainable={info['trainable_parameters']})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()


# Compatibility alias for backward compatibility
BaseModel = BaseAnomalyDetectionModel
