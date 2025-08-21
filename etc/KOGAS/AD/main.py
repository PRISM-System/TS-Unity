"""
Anomaly Detection Main Module

이 모듈은 이상 탐지 모델의 학습, 테스트, 추론을 위한 메인 실행 파일입니다.
"""

import wandb
import torch
import yaml
import json
import os
import warnings
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf

from utils.utils import set_seed, log_setting, version_build
from data_provider.dataloader import get_dataloader
from model import AnomalyDetectionModel

warnings.filterwarnings('ignore')


class AnomalyDetectionRunner:
    """이상 탐지 모델 실행을 위한 메인 클래스"""
    
    def __init__(self, args: argparse.Namespace, config: Dict[str, Any]):
        """
        Args:
            args: 명령행 인수
            config: 설정 파일 내용
        """
        self.args = args
        self.config = config
        self.logger = None
        self.savedir = None
        
    def setup_logging(self) -> None:
        """로깅 설정을 초기화합니다."""
        now = datetime.now()
        
        # 프로세스 이름 생성
        if self.args.subdataname is not None:
            group_name = f'{self.args.dataname}/{self.args.subdataname}-{self.args.model}'
        else:
            group_name = f'{self.args.dataname}-{self.args.model}'
            
        process_name = f'{group_name}-{now.strftime("%Y%m%d_%H%m%S")}'
        
        # 저장 디렉토리 설정
        if self.args.subdataname is not None:
            logdir = os.path.join(
                self.config['log_dir'], 
                f'{self.args.dataname}/{self.args.model}/{self.args.subdataname}'
            )
        else:
            logdir = os.path.join(
                self.config['log_dir'], 
                f'{self.args.dataname}/{self.args.model}'
            )
            
        self.savedir = version_build(
            logdir=logdir, 
            is_train=self.args.train, 
            resume=self.args.resume
        )
        
        self.logger = log_setting(
            self.savedir, 
            f'{now.strftime("%Y%m%d_%H%m%S")}'
        )
        
        # 인수 저장
        json.dump(
            vars(self.args), 
            open(os.path.join(self.savedir, 'arguments.json'), 'w'), 
            indent=4
        )
        
        self.logger.info(f'Process {process_name} start!')
        
    def setup_gpu(self) -> None:
        """GPU 설정을 초기화합니다."""
        # CUDA 사용 가능 여부 확인
        if self.args.use_gpu and not torch.cuda.is_available():
            self.logger.warning("CUDA가 사용 불가능합니다. CPU로 자동 전환됩니다.")
            self.args.use_gpu = False
            
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
            
    def setup_wandb(self, sweep_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Weights & Biases 설정을 초기화합니다."""
        if not self.args.log_to_wandb:
            model_params = self.config[self.args.model]
            scale = self.config['scale']
            return model_params, scale
            
        now = datetime.now()
        if self.args.subdataname is not None:
            group_name = f'{self.args.dataname}/{self.args.subdataname}-{self.args.model}'
        else:
            group_name = f'{self.args.dataname}-{self.args.model}'
            
        process_name = f'{group_name}-{now.strftime("%Y%m%d_%H%m%S")}'
        
        if self.args.sweep:
            wandb.init(
                name=process_name,
                group=group_name,
                project='variable-transformer',
                config=sweep_config
            )
            self.logger.info('wandb sweep init success!!')
            model_params = wandb.config
            scale = model_params['scale']
        else:
            wandb.init(
                name=process_name,
                group=group_name,
                project='variable-transformer',
                config=self.config
            )
            self.logger.info('wandb init success!!')
            model_params = self.config[self.args.model]
            scale = self.config['scale']
            
        return model_params, scale
        
    def get_model_type(self) -> str:
        """모델 타입을 결정합니다."""
        reconstruction_models = {
            'VTTSAT', 'VTTPAT', 'LSTM_AE', 'LSTM_VAE', 'TF', 
            'USAD', 'OmniAnomaly', 'DAGMM', 'AnomalyTransformer'
        }
        
        if self.args.model in reconstruction_models:
            return 'reconstruction'
        elif self.args.model in (''):  # 예측 모델들
            return 'prediction'
        else:
            return 'classification'
            
    def load_data(self, data_info: Any, scale: str, model_type: str) -> tuple:
        """데이터를 로드합니다."""
        return get_dataloader(
            data_name=self.args.dataname,
            sub_data_name=self.args.subdataname,
            data_info=data_info,
            loader_params=self.config['loader_params'],
            scale=scale,
            window_size=self.args.window_size,
            slide_size=int(self.args.slide_size),
            model_type=model_type
        )
        
    def run_training(self, model: Any, trainloader: Any, validloader: Any, testloader: Any) -> None:
        """모델 학습을 실행합니다."""
        if not self.args.train:
            return
            
        self.logger.info('Training Start!')
        history = model.train(trainloader, validloader, testloader)
        
        for i, (train_loss, valid_loss) in enumerate(
            zip(history['train_loss'], history['validation_loss'])
        ):
            self.logger.info(
                f"Epoch: {i + 1} Train Loss: {train_loss:.7f} Vali Loss: {valid_loss:.7f}"
            )
        self.logger.info('Model training success!!')
        
    def run_testing(self, model: Any, validloader: Any, testloader: Any) -> None:
        """모델 테스트를 실행합니다."""
        if not self.args.test:
            return
            
        K = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        history, auc = model.test(validloader, testloader)
        self.logger.info(f"PA%K AUC: {auc:.4f}")
        self.logger.info('Model test success!!')
        
    def run_inference(self, model: Any, testloader: Any, validloader: Any) -> None:
        """모델 추론을 실행합니다."""
        if not self.args.inference:
            return
            
        model.inference(testloader, valid_loader=validloader)
        self.logger.info('Model inference success!!')
        
    def run(self, sweep_config: Optional[Dict[str, Any]] = None) -> None:
        """메인 실행 함수입니다."""
        try:
            # 초기 설정
            self.setup_logging()
            self.setup_gpu()
            set_seed(self.args.seed)
            
            # wandb 설정
            model_params, scale = self.setup_wandb(sweep_config)
            
            # 모델 파라미터 저장
            json.dump(
                dict(model_params), 
                open(os.path.join(self.savedir, 'model_params.json'), 'w'), 
                indent=4
            )
            
            # 모델 타입 결정
            model_type = self.get_model_type()
            
            # 데이터 정보 로드
            data_info = OmegaConf.create(self.config[self.args.dataname])
            
            # 데이터 로드
            trainloader, validloader, testloader, interloader = self.load_data(
                data_info, scale, model_type
            )
            
            # 데이터 형태 확인
            model_params['window_size'] = self.args.window_size
            b, model_params['window_size'], model_params['feature_num'] = next(
                iter(trainloader)
            )['given'].shape
            self.logger.info(
                f'Data shape is ({b}, {model_params.window_size}, {model_params.feature_num})'
            )
            
            # 모델 빌드
            model = AnomalyDetectionModel(self.args, model_params, self.savedir)
            # logger 설정
            model.set_logger(self.logger)
            self.logger.info('Model build success!!')
            
            if self.args.log_to_wandb:
                wandb.watch(model.model)
                
            # 실행
            self.run_training(model, trainloader, validloader, testloader)
            self.run_testing(model, validloader, testloader)
            self.run_inference(model, testloader, validloader)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during execution: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()


def create_argument_parser() -> argparse.ArgumentParser:
    """명령행 인수 파서를 생성합니다."""
    parser = argparse.ArgumentParser(description='Anomaly Detection Model Runner')
    
    # 실행 모드
    parser.add_argument('--train', action='store_true', help='모델 학습')
    parser.add_argument('--test', action='store_true', help='이상 점수 계산')
    parser.add_argument('--inference', action='store_true', help='추론만 실행')
    parser.add_argument('--sweep', action='store_true', help='wandb로 하이퍼파라미터 탐색')
    parser.add_argument('--resume', type=int, default=None, help='재학습 또는 테스트할 버전 번호')
    parser.add_argument('--init_weight', type=str, default=None, help='전이학습을 위한 초기 가중치 디렉토리')
    
    # 모델 선택
    parser.add_argument('--model', type=str, required=True,
                        choices=['VTTSAT', 'VTTPAT', 'InterFusion', 'OmniAnomaly', 'LSTM_AE', 
                                 'LSTM_VAE', 'DAGMM', 'TF', 'USAD', 'AnomalyTransformer'],
                        help="모델 이름")
    
    # 데이터 설정
    parser.add_argument('--dataname', type=str, required=True, 
                        choices=['SWaT', 'SKAB', 'KOGAS', 'KOGAS2', 'KOGAS3'],
                        help='데이터셋 이름')
    parser.add_argument('--subdataname', type=str, default=None, help='서브 데이터셋 이름')
    parser.add_argument('--window_size', type=int, default=100, help='데이터 로더용 윈도우 크기')
    parser.add_argument('--slide_size', type=int, default=50, help='데이터 로더용 오버랩 비율')
    parser.add_argument('--train_path', type=str, default=None, help='학습 데이터 경로')
    parser.add_argument('--test_path', type=str, default=None, help='테스트 데이터 경로')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=30, help='에포크 수')
    parser.add_argument('--patience', type=int, default=5, help='조기 종료 인내심')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    
    # 손실 함수
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'],
                        help='손실 함수 선택')
    
    # 시스템 설정
    parser.add_argument('--seed', type=int, default=72, help="랜덤 시드")
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, type=bool, 
                        default=torch.cuda.is_available(), help='GPU 사용 여부 (CUDA 사용 불가시 자동으로 CPU 사용)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 번호')
    parser.add_argument('--use_multi_gpu', action='store_true', help='멀티 GPU 사용', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='멀티 GPU 디바이스 ID')
    
    # 로깅 설정
    parser.add_argument('--log_to_wandb', action='store_true', default=False, help='wandb 로깅 사용')
    parser.add_argument('--configure', type=str, default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--sweep_count', type=int, default=30, help='하이퍼파라미터 탐색 횟수')
    
    return parser


def load_config(config_path: str, is_sweep: bool = False) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    with open(config_path) as f:
        if is_sweep:
            return yaml.safe_load(f)
        else:
            return OmegaConf.load(f)


def main(sweep_config: Optional[Dict[str, Any]] = None) -> None:
    """
    메인 실행 함수
    
    Args:
        sweep_config: wandb sweep 설정 (선택사항)
    """
    # 인수 파싱
    parser = create_argument_parser()
    args = parser.parse_args()
    print(args)
    
    # 설정 파일 로드
    config = load_config(args.configure, args.sweep)
    
    # 실행기 생성 및 실행
    runner = AnomalyDetectionRunner(args, config)
    
    if args.sweep:
        sweep_id = wandb.sweep(config, project="variable-transformer", entity='hwk0702')
        wandb.agent(sweep_id, main, count=args.sweep_count)
    else:
        runner.run(sweep_config)


if __name__ == '__main__':
    main()