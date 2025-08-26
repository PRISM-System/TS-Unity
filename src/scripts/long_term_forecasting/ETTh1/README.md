# DLinear ETTh1 Long-term Forecasting Experiments

이 폴더는 ETTh1 데이터셋을 사용한 DLinear 모델의 long-term forecasting 실험을 위한 스크립트들을 포함합니다.

## 📁 파일 구조

```
ETTh1/
├── DLinear_ETTh1_96_96.sh      # 96→96 예측 실험
├── DLinear_ETTh1_96_192.sh     # 96→192 예측 실험  
├── DLinear_ETTh1_96_336.sh     # 96→336 예측 실험
├── DLinear_ETTh1_96_720.sh     # 96→720 예측 실험
├── DLinear_ETTh1_test.sh       # 모든 모델 테스트
├── run_all_DLinear_ETTh1.sh    # 전체 실험 실행
└── README.md                    # 이 파일
```

## 🚀 사용법

### 1. 개별 실험 실행

특정 예측 길이의 실험만 실행하려면:

```bash
# 96→96 예측 실험
./DLinear_ETTh1_96_96.sh

# 96→192 예측 실험
./DLinear_ETTh1_96_192.sh

# 96→336 예측 실험
./DLinear_ETTh1_96_336.sh

# 96→720 예측 실험
./DLinear_ETTh1_96_720.sh
```

### 2. 전체 실험 실행

모든 예측 길이의 실험을 순차적으로 실행하려면:

```bash
./run_all_DLinear_ETTh1.sh
```

### 3. 테스트만 실행

훈련된 모델들을 테스트하려면:

```bash
./DLinear_ETTh1_test.sh
```

## ⚙️ 실험 설정

### 기본 하이퍼파라미터
- **모델**: DLinear
- **데이터**: ETTh1 (ETT-small)
- **입력 길이**: 96
- **레이블 길이**: 48
- **예측 길이**: 96, 192, 336, 720
- **특성**: Multivariate (M)
- **배치 크기**: 32
- **학습률**: 0.0001
- **에포크**: 100
- **Early Stopping**: 10

### 모델 아키텍처
- **Encoder Layers**: 2
- **Decoder Layers**: 1
- **Factor**: 3
- **Input/Output Dimensions**: 7
- **Model Dimension**: 512
- **Attention Heads**: 8
- **Feed-forward Dimension**: 2048
- **Moving Average Window**: 25

## 📊 예상 결과

각 실험은 다음 메트릭들을 제공합니다:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **MSPE** (Mean Squared Percentage Error)

## 📁 결과 저장 위치

- **훈련 로그**: `./checkpoints/logs/`
- **모델 체크포인트**: `./checkpoints/DLinear_long_term_forecast_ETTh1_*/`
- **테스트 결과**: `./src/results/test/`

## 🔧 문제 해결

### 권한 문제
스크립트 실행 권한이 없다면:
```bash
chmod +x *.sh
```

### GPU 메모리 부족
`CUDA_VISIBLE_DEVICES`를 다른 GPU로 변경하거나, `batch_size`를 줄이세요.

### 데이터 경로 문제
ETTh1 데이터가 `./datasets/ETT-small/` 경로에 있는지 확인하세요.

## 📚 참고 자료

- **DLinear 논문**: [Time Series Decomposition Transformer](https://arxiv.org/pdf/2205.13504.pdf)
- **ETT 데이터셋**: [Electricity Transformer Temperature](https://github.com/thuml/Time-Series-Library)

## 🤝 기여

이 스크립트들을 수정하거나 개선하고 싶다면, 새로운 설정이나 하이퍼파라미터를 추가하여 pull request를 보내주세요.
