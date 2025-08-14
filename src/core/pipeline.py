"""
Training Pipeline Module

This module provides a comprehensive training pipeline for time series analysis tasks,
supporting multiple task types and models with configurable training parameters.
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path
import argparse
import json
import logging
from dataclasses import dataclass

from config.base_config import (
    BaseConfig, ForecastingConfig, AnomalyDetectionConfig, 
    ImputationConfig, ClassificationConfig
)
from core.base_trainer import BaseTrainer, ForecastingTrainer
from data_provider.data_factory import DataFactory
from utils.logger import ExperimentLogger
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for different task types."""
    name: str
    exp_class: Type
    config_class: Type[BaseConfig]


class TaskRegistry:
    """Registry for available tasks and their configurations."""
    
    TASKS = {
        'long_term_forecast': TaskConfig(
            name='long_term_forecast',
            exp_class=Exp_Long_Term_Forecast,
            config_class=ForecastingConfig
        ),
        'short_term_forecast': TaskConfig(
            name='short_term_forecast',
            exp_class=Exp_Short_Term_Forecast,
            config_class=ForecastingConfig
        ),
        'anomaly_detection': TaskConfig(
            name='anomaly_detection',
            exp_class=Exp_Anomaly_Detection,
            config_class=AnomalyDetectionConfig
        ),
        'imputation': TaskConfig(
            name='imputation',
            exp_class=Exp_Imputation,
            config_class=ImputationConfig
        ),
        'classification': TaskConfig(
            name='classification',
            exp_class=Exp_Classification,
            config_class=ClassificationConfig
        )
    }
    
    @classmethod
    def get_task_config(cls, task_name: str) -> Optional[TaskConfig]:
        """Get task configuration by name."""
        return cls.TASKS.get(task_name)
    
    @classmethod
    def get_available_tasks(cls) -> list:
        """Get list of available task names."""
        return list(cls.TASKS.keys())
    
    @classmethod
    def get_config_class(cls, task_name: str) -> Optional[Type[BaseConfig]]:
        """Get configuration class for a task."""
        task_config = cls.get_task_config(task_name)
        return task_config.config_class if task_config else None


class TrainingPipeline:
    """Main training pipeline for time series analysis tasks."""
    
    def __init__(self, config: BaseConfig, use_wandb: bool = False):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.use_wandb = use_wandb
        self.logger: Optional[ExperimentLogger] = None
        
        self._setup_pipeline()
    
    def _setup_pipeline(self) -> None:
        """Setup the pipeline components."""
        self._set_seed(self.config.seed)
        self._setup_experiment()
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        logger.info(f"Random seed set to: {seed}")
    
    def _setup_experiment(self) -> None:
        """Setup experiment logging and configuration."""
        experiment_name = self._generate_experiment_name()
        
        self.logger = ExperimentLogger(
            name=experiment_name,
            log_dir=f"{self.config.checkpoints}/logs",
            use_wandb=self.use_wandb,
            wandb_project="ts-unity",
            wandb_config=self.config.to_dict()
        )
        
        self.logger.log_config(self.config.to_dict())
        self.logger.log_system_info()
        
        logger.info(f"Experiment setup complete: {experiment_name}")
    
    def _generate_experiment_name(self) -> str:
        """Generate experiment name from configuration."""
        return f"{self.config.model}_{self.config.task_name}_{self.config.data}_{self.config.des}"
    
    def create_experiment(self):
        """Create experiment instance based on task configuration."""
        task_config = TaskRegistry.get_task_config(self.config.task_name)
        
        if task_config is None:
            raise ValueError(f"Unsupported task: {self.config.task_name}. "
                           f"Available tasks: {TaskRegistry.get_available_tasks()}")
        
        logger.info(f"Creating experiment for task: {self.config.task_name}")
        return task_config.exp_class(self.config)
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the training process.
        
        Returns:
            Dictionary containing training results
        """
        exp = self.create_experiment()
        self.logger.log_model_info(exp.model)
        
        if not self.config.is_training:
            logger.info("Training mode disabled, running test only")
            return self._run_test(exp)
        
        results = {}
        logger.info(f"Starting training for {self.config.itr} iterations")
        
        for iteration in range(self.config.itr):
            logger.info(f"Training iteration {iteration + 1}/{self.config.itr}")
            
            # Training phase
            exp.train()
            
            # Testing phase
            test_results = self._run_test(exp)
            results[f'iteration_{iteration + 1}'] = test_results
            
            # Cleanup
            torch.cuda.empty_cache()
        
        # Log final results
        self.logger.log_metrics(test_results, prefix="final_test_")
        logger.info("Training completed successfully")
        
        return results
    
    def _run_test(self, exp) -> Dict[str, Any]:
        """Run testing phase."""
        logger.info("Running test phase")
        test_results = exp.test()
        self.logger.log_metrics(test_results, prefix="test_")
        return test_results
    
    def run_prediction(self) -> Dict[str, Any]:
        """
        Run prediction phase.
        
        Returns:
            Dictionary containing prediction results
        """
        exp = self.create_experiment()
        
        logger.info("Running prediction phase")
        pred_results = exp.predict()
        
        return pred_results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.logger:
            self.logger.close()
        logger.info("Pipeline closed")


class ConfigManager:
    """Manages configuration creation and validation."""
    
    @staticmethod
    def create_config_from_args(args: argparse.Namespace) -> BaseConfig:
        """Create configuration object from command line arguments."""
        config_class = TaskRegistry.get_config_class(args.task_name)
        
        if config_class is None:
            raise ValueError(f"Invalid task name: {args.task_name}")
        
        # Filter out None values
        config_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        return config_class(**config_dict)
    
    @staticmethod
    def setup_args() -> argparse.Namespace:
        """Setup command line argument parser."""
        parser = argparse.ArgumentParser(description='Time Series Analysis Pipeline')
        
        # Task configuration
        parser.add_argument(
            '--task_name', type=str, required=True, default='long_term_forecast',
            choices=TaskRegistry.get_available_tasks(),
            help='Task name for time series analysis'
        )
        parser.add_argument(
            '--is_training', type=int, required=True, default=1,
            help='Training mode (1: training, 0: testing only)'
        )
        
        # Model configuration
        parser.add_argument(
            '--model', type=str, required=True, default='Autoformer',
            help='Model name for time series analysis'
        )
        
        # Data configuration
        parser.add_argument(
            '--data', type=str, required=True, default='ETTh1',
            help='Dataset type'
        )
        parser.add_argument(
            '--root_path', type=str, default='./data/ETT/',
            help='Root path of the data file'
        )
        parser.add_argument(
            '--data_path', type=str, default='ETTh1.csv',
            help='Data file name'
        )
        parser.add_argument(
            '--features', type=str, default='M',
            choices=['M', 'S', 'MS'],
            help='Forecasting task type: M (multivariate), S (univariate), MS (multivariate->univariate)'
        )
        parser.add_argument(
            '--target', type=str, default='OT',
            help='Target feature for S or MS tasks'
        )
        parser.add_argument(
            '--freq', type=str, default='h',
            help='Frequency for time features encoding'
        )
        parser.add_argument(
            '--checkpoints', type=str, default='./checkpoints/',
            help='Location of model checkpoints'
        )
        
        # Sequence configuration
        parser.add_argument(
            '--seq_len', type=int, default=96,
            help='Input sequence length'
        )
        parser.add_argument(
            '--label_len', type=int, default=48,
            help='Start token length'
        )
        parser.add_argument(
            '--pred_len', type=int, default=96,
            help='Prediction sequence length'
        )
        parser.add_argument(
            '--seasonal_patterns', type=str, default='Monthly',
            help='Seasonal patterns for M4 dataset'
        )
        
        # Model architecture
        parser.add_argument(
            '--enc_in', type=int, default=7,
            help='Encoder input size'
        )
        parser.add_argument(
            '--dec_in', type=int, default=7,
            help='Decoder input size'
        )
        parser.add_argument(
            '--c_out', type=int, default=7,
            help='Output size'
        )
        parser.add_argument(
            '--d_model', type=int, default=512,
            help='Dimension of model'
        )
        parser.add_argument(
            '--n_heads', type=int, default=8,
            help='Number of attention heads'
        )
        parser.add_argument(
            '--e_layers', type=int, default=2,
            help='Number of encoder layers'
        )
        parser.add_argument(
            '--d_layers', type=int, default=1,
            help='Number of decoder layers'
        )
        parser.add_argument(
            '--d_ff', type=int, default=2048,
            help='Dimension of feed-forward network'
        )
        parser.add_argument(
            '--moving_avg', type=int, default=25,
            help='Window size of moving average'
        )
        parser.add_argument(
            '--factor', type=int, default=1,
            help='Attention factor'
        )
        parser.add_argument(
            '--distil', action='store_false', default=True,
            help='Whether to use distilling in encoder'
        )
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='Dropout rate'
        )
        parser.add_argument(
            '--embed', type=str, default='timeF',
            choices=['timeF', 'fixed', 'learned'],
            help='Time features encoding method'
        )
        parser.add_argument(
            '--activation', type=str, default='gelu',
            help='Activation function'
        )
        parser.add_argument(
            '--output_attention', action='store_true',
            help='Whether to output attention in encoder'
        )
        
        # Training configuration
        parser.add_argument(
            '--num_workers', type=int, default=10,
            help='Data loader number of workers'
        )
        parser.add_argument(
            '--itr', type=int, default=1,
            help='Number of experiment iterations'
        )
        parser.add_argument(
            '--train_epochs', type=int, default=10,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch_size', type=int, default=32,
            help='Training batch size'
        )
        parser.add_argument(
            '--patience', type=int, default=3,
            help='Early stopping patience'
        )
        parser.add_argument(
            '--learning_rate', type=float, default=0.0001,
            help='Learning rate'
        )
        parser.add_argument(
            '--des', type=str, default='test',
            help='Experiment description'
        )
        parser.add_argument(
            '--loss', type=str, default='MSE',
            help='Loss function'
        )
        parser.add_argument(
            '--lradj', type=str, default='type1',
            help='Learning rate adjustment strategy'
        )
        parser.add_argument(
            '--use_amp', action='store_true', default=False,
            help='Use automatic mixed precision training'
        )
        
        # Hardware configuration
        parser.add_argument(
            '--use_gpu', type=bool, default=True,
            help='Whether to use GPU'
        )
        parser.add_argument(
            '--gpu', type=int, default=0,
            help='GPU device ID'
        )
        parser.add_argument(
            '--gpu_type', type=str, default='cuda',
            help='GPU type (cuda, mps)'
        )
        parser.add_argument(
            '--use_multi_gpu', action='store_true', default=False,
            help='Use multiple GPUs'
        )
        parser.add_argument(
            '--devices', type=str, default='0,1,2,3',
            help='Device IDs for multi-GPU setup'
        )
        parser.add_argument(
            '--seed', type=int, default=2021,
            help='Random seed'
        )
        
        # Additional options
        parser.add_argument(
            '--use_dtw', type=bool, default=False,
            help='Use DTW or not'
        )
        parser.add_argument(
            '--use_tqdm', type=bool, default=True,
            help='Use tqdm progress bar'
        )
        parser.add_argument(
            '--use_wandb', action='store_true', default=False,
            help='Use Weights & Biases for logging'
        )
        parser.add_argument(
            '--config_file', type=str,
            help='Path to configuration file'
        )
        
        args = parser.parse_args()
        
        # Handle configuration file
        if args.config_file:
            config = BaseConfig.from_yaml(args.config_file)
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None:
                    setattr(config, key, value)
            return config
        
        return ConfigManager.create_config_from_args(args)


def main() -> None:
    """Main entry point for the training pipeline."""
    try:
        args = ConfigManager.setup_args()
        pipeline = TrainingPipeline(args, use_wandb=args.use_wandb)
        
        if args.is_training:
            results = pipeline.run_training()
            logger.info(f"Training completed successfully. Results: {results}")
        else:
            results = pipeline.run_prediction()
            logger.info(f"Prediction completed successfully. Results: {results}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise
    finally:
        if 'pipeline' in locals():
            pipeline.close()


if __name__ == '__main__':
    main()