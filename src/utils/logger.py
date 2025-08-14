import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
import torch
import psutil
import wandb


class ExperimentLogger:
    def __init__(
        self,
        name: str,
        log_dir: str = "./logs",
        level: int = logging.INFO,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project or "ts-unity",
                name=name,
                config=wandb_config
            )
        
        self.setup_logger(level)
        self.metrics_history = []
        self.start_time = time.time()
        
    def setup_logger(self, level: int):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        self.metrics_history.append(log_entry)
        
        metric_str = " - ".join([f"{prefix}{k}: {v:.6f}" for k, v in metrics.items()])
        step_str = f"Step {step} - " if step is not None else ""
        self.logger.info(f"{step_str}{metric_str}")
        
        if self.use_wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)
    
    def log_model_info(self, model: torch.nn.Module, input_size: Optional[tuple] = None):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_name': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
        if input_size:
            model_info['input_size'] = input_size
        
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
        
        if self.use_wandb:
            wandb.log(model_info)
    
    def log_system_info(self):
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_count': torch.cuda.device_count()
            })
        
        self.logger.info(f"System Info: {json.dumps(system_info, indent=2)}")
        
        if self.use_wandb:
            wandb.log(system_info)
    
    def log_config(self, config: Dict[str, Any]):
        config_path = self.log_dir / f"{self.name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_path}")
        
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        elapsed_time = time.time() - self.start_time
        epoch_summary = {
            'epoch': epoch,
            'elapsed_time': elapsed_time,
            'learning_rate': lr,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        self.logger.info(f"Epoch {epoch} Summary: {json.dumps(epoch_summary, indent=2)}")
        
        if self.use_wandb:
            wandb.log(epoch_summary, step=epoch)
    
    def save_metrics_history(self):
        metrics_path = self.log_dir / f"{self.name}_metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        self.logger.info(f"Metrics history saved to {metrics_path}")
    
    def close(self):
        self.save_metrics_history()
        if self.use_wandb:
            wandb.finish()
        
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }
    
    def get_totals(self) -> Dict[str, float]:
        return self.metrics.copy()


def setup_distributed_logging(rank: int, world_size: int):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    logger = logging.getLogger(f"rank_{rank}")
    logger.info(f"Process {rank}/{world_size} initialized")
    return logger