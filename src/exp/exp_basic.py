import os
import torch
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer
from core.base_model import BaseTimeSeriesModel
from config.base_config import BaseConfig


class Exp_Basic(ABC):
    def __init__(self, args: BaseConfig):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.logger.info(f'Model initialized: {self.model.__class__.__name__}')
        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')

    @abstractmethod
    def _build_model(self) -> BaseTimeSeriesModel:
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    @abstractmethod
    def _get_data(self):
        raise NotImplementedError

    @abstractmethod
    def vali(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dict[str, float]:
        raise NotImplementedError
        
    def setup_logging(self):
        log_dir = Path(self.args.checkpoints) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'{self.args.model}_{self.args.task_name}.log'),
                logging.StreamHandler()
            ]
        )
        
    def get_model_info(self) -> Dict[str, Any]:
        if hasattr(self, 'model'):
            return self.model.get_model_info()
        return {}
        
    def save_config(self, path: Optional[str] = None):
        if path is None:
            config_dir = Path(self.args.checkpoints) / self.args.des
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / 'config.yaml'
        
        self.args.save_yaml(str(path))
        self.logger.info(f'Config saved to {path}')
