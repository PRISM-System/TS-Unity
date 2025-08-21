# TS-Unity Inference Guide (DB → NumPy → Inference)

API 없이 DB(또는 파일)에서 읽은 데이터를 바로 추론에 사용하는 경우를 위한 가이드입니다. 입력 데이터 형식, 함수별 출력/리턴 스펙, 슬라이딩 윈도우/배치/스트리밍 사용법을 정리합니다.

## 공통 전제
- 입력 타입: `numpy.ndarray` (float32 권장)
- 컬럼 순서/스케일링: 학습 시 사용한 전처리(정규화, 컬럼 순서)를 동일하게 적용
- 피쳐 수: `config.enc_in`
- 시퀀스 길이: `config.seq_len`

## 데이터 스키마/이름 규칙 (중요)
DB에서 읽은 테이블/뷰를 추론 입력으로 사용할 때는 **열 이름과 순서**를 명확히 정의해야 합니다.

- **feature_names (필수 권장)**: 모델 입력에 사용할 컬럼 이름의 **순서 있는 리스트**
  - 길이 = `enc_in`
  - 이 순서대로 DataFrame을 슬라이싱하여 `(…, enc_in)` 배열을 만듭니다.
- **target (Forecasting S/MS에서 중요)**:
  - `features='M'`: 다변량→다변량. 일반적으로 `c_out == enc_in` (전체 컬럼 예측)
  - `features='MS'`: 다변량→단변량. `target` 컬럼 1개를 예측 (`c_out == 1`)
  - `features='S'`: 단변량→단변량. 입력/출력 모두 `target` 1개 (`enc_in == 1`, `c_out == 1`)
- **timestamp/time 컬럼**: 정렬용으로만 사용하고 **모델 입력 배열에는 포함하지 않는 것**을 권장
- **config.data**: 데이터셋 식별용 문자열(태깅)이며, DB 스키마와 직접적 제약은 없음

### 예시: DB → DataFrame → NumPy 매핑
```python
# 예) DataFrame df, 열: ['timestamp','load','temp','humid','wind','pressure','solar']
feature_names = ['load','temp','humid','wind','pressure','solar']  # enc_in=6
X = df.sort_values('timestamp')[feature_names].values.astype('float32')  # (T, 6)
```

### Forecasting 모드별 설정/형상
| features 모드 | 입력(enc_in)       | 출력(c_out)       | 비고 |
|--------------|---------------------|-------------------|------|
| M            | 다변량(=K개)         | 다변량(=K개)       | 전체 컬럼 예측 |
| MS           | 다변량(=K개)         | 단변량(=1개)       | `target` 지정 필수 |
| S            | 단변량(=1개)         | 단변량(=1개)       | `feature_names=[target]` |

### 검증 스니펫
```python
def validate_schema(df, feature_names, enc_in):
    # 1) 컬럼 존재 검증
    missing = [c for c in feature_names if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    # 2) 순서/길이 검증
    assert len(feature_names) == enc_in, f"enc_in={enc_in}, but got {len(feature_names)} features"
    # 3) 결측/타입 처리 (예시)
    X = df[feature_names].astype('float32').fillna(method='ffill').fillna(method='bfill').values
    return X
```

### Anomaly Detection 데이터 이름 규칙
- 일반적으로 **모든 센서 채널**을 feature로 사용 (다변량 입력)
- `feature_names` 길이는 `enc_in`과 동일
- `target` 없음 (스코어만 산출)

---

## 입력 형식
- 단일 시퀀스: `(seq_len, num_features)` = `(config.seq_len, config.enc_in)`
- 배치 시퀀스: `(batch_size, seq_len, num_features)`
- 스트리밍 단일 포인트: `(num_features,)`
- 슬라이딩 윈도우 원본: `(time_steps, num_features)` (time_steps ≥ window_size)

## 핵심 객체
- `InferencePipeline(config, checkpoint_path)`
  - 모델/태스크에 맞는 추론 파이프라인. 주요 메서드:
    - `predict_batch(input_data, num_steps=1)`
    - `predict_next(num_steps)`
    - `predict_with_sliding_window(input_data, window_size=None, stride=1, num_steps=1)`
    - (선택) `predict_from_file(file_path, num_steps=1, output_path=None)`

---

## Forecasting (예측)
미래 `num_steps` 스텝 예측.

### 출력/리턴 스펙
- `predict_batch(input_data, num_steps)`
  - 입력: `(seq_len, C)` 또는 `(B, seq_len, C)`
  - 출력: `(1, num_steps, C)` 또는 `(B, num_steps, C)`
- `predict_next(num_steps)`
  - 입력: 내부 버퍼가 `config.seq_len`만큼 채워진 상태
  - 출력: `(num_steps, C)`
- `predict_with_sliding_window(input_data, window_size, stride, num_steps)`
  - 입력: `(T, C)`
  - 리턴: dict
    - `predictions`: `(N, 1, num_steps, C)`
    - `input_windows`: `(N, window_size, C)`
    - `window_indices`: `List[(start_idx, end_idx)]`
    - `num_windows`, `window_size`, `stride`, `num_steps`, `task_type`

### 간단 예시
```python
import numpy as np
from core.pipeline import InferencePipeline
from config.base_config import ForecastingConfig

config = ForecastingConfig(task_name='long_term_forecast', model='Autoformer',
                           seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7)
pipeline = InferencePipeline(config, checkpoint_path='checkpoints/autoformer.pth')

sequence = np.asarray(sequence, dtype='float32')  # (96, 7)
pred = pipeline.predict_batch(sequence, num_steps=24)  # (1, 24, 7)

long_series = np.asarray(long_series, dtype='float32')  # (T, 7)
res = pipeline.predict_with_sliding_window(long_series, window_size=96, stride=10, num_steps=24)
# res['predictions'] shape: (N, 1, 24, 7)
```

---

## Anomaly Detection (이상탐지)
모델 유형에 따라 자동 선택:
- **Reconstruction-based** (AnomalyTransformer, USAD, DAGMM, AE/VAE 등)
  - 스코어: 재구성 오차 (클수록 이상)
- **Prediction-based** (Autoformer/Transformer 등 예측 모델)
  - 스코어: 예측 패턴(분산/오차) (클수록 이상)

### 출력/리턴 스펙
- `predict_batch(input_data, num_steps=1)`  ← num_steps는 무시됨
  - 입력: `(seq_len, C)` 또는 `(B, seq_len, C)`
  - 출력(점수):
    - Reconstruction-based: `(B, seq_len)`  # 시점별 스코어
    - Prediction-based: `(B, 1, C)`         # 피쳐별 스코어
- `predict_with_sliding_window(input_data, window_size, stride, num_steps=1)`
  - 입력: `(T, C)`
  - 리턴: dict
    - `predictions`: 윈도우 축 N으로 누적된 점수 텐서 (위 `predict_batch` 출력 형태가 N개)
    - `input_windows`: `(N, window_size, C)`
    - `window_indices`, `num_windows`, `window_size`, `stride`, `num_steps(=1)`, `task_type`
    - `detection_method`: `{ method: 'reconstruction_based'|'prediction_based', model_type, description, approach }`

### 간단 예시
```python
import numpy as np
from core.pipeline import InferencePipeline
from config.base_config import AnomalyDetectionConfig, ForecastingConfig

# Reconstruction-based
cfg_rec = AnomalyDetectionConfig(task_name='anomaly_detection', model='AnomalyTransformer',
                                 seq_len=100, enc_in=7, dec_in=7, c_out=7)
pip_rec = InferencePipeline(cfg_rec, checkpoint_path='checkpoints/anom_trans.pth')
seq = np.asarray(seq, dtype='float32')  # (100, 7)
scores_rec = pip_rec.predict_batch(seq)  # (1, 100)

# Prediction-based
cfg_pred = ForecastingConfig(task_name='anomaly_detection', model='Autoformer',
                             seq_len=96, pred_len=24, enc_in=7, dec_in=7, c_out=7)
pip_pred = InferencePipeline(cfg_pred, checkpoint_path='checkpoints/autoformer.pth')
seq2 = np.asarray(seq2, dtype='float32')  # (96, 7)
scores_pred = pip_pred.predict_batch(seq2)  # (1, 1, 7)

series = np.asarray(series, dtype='float32')
res = pip_pred.predict_with_sliding_window(series, window_size=96, stride=10)
method = res.get('detection_method', {})
```

---

## 파일 기반 추론 (선택)
- `predict_from_file(file_path, num_steps=1, output_path=None)`
  - 리턴: dict
    - `predictions`: Forecasting → `(N, 1, num_steps, C)`, Anomaly → `(N, …)`
    - `input_windows`: `(N, window_size, C)`
    - `num_windows`, `num_steps(또는 1)`, `input_shape`, `task_type`
    - `detection_method` (anomaly일 때)

---

## 리턴 스키마 요약
- **Forecasting**
  - 텐서: `(B, num_steps, C)` 또는 `(1, num_steps, C)`
  - 슬라이딩: dict(`predictions` = `(N, 1, num_steps, C)`, 기타 메타)
- **Anomaly Detection**
  - Reconstruction-based: `(B, seq_len)`
  - Prediction-based: `(B, 1, C)`
  - 슬라이딩: dict(`predictions` = 위 점수 텐서가 윈도우 축 `N`으로 누적, `detection_method` 포함)

---

## DB 연동 팁
- DB→DataFrame→`values.astype('float32')` 변환 후 바로 입력
- 시간 오름차순 정렬 및 결측치 처리 후 투입 권장
- 학습 시 사용한 스케일러(예: StandardScaler)로 동일 변환 적용

## 검증 체크리스트
- `input.shape[1] == config.enc_in`
- (배치 제외) `input.shape[-2] == config.seq_len`
- 학습 시 피쳐 순서/정규화와 동일 여부 