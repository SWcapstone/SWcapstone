# Gate Model (MobileNetV3-Small) - 데이터 분류 및 학습 결과

브랜치: `experiment/mnv3-small-cascade`
작성일: 2026-05-18
모델: MobileNetV3-Small (927K params)
데이터셋: KolektorSDD2 (표면 결함 검출)

---

# 1. 데이터셋 구성

원본 KolektorSDD2 데이터를 3단계 증강(augmentation) 수준으로 나눠 v1/v2/v3 버전을 구성했습니다.

## 1-1. 버전별 전체 데이터 규모

| Version | 증강 수준 | 총 이미지 | Normal | Anomaly | Anomaly 비율 |
|---------|----------|----------|--------|---------|-------------|
| v1 | 기본 증강 (aug x3) | 12,976 | 11,868 | 1,108 | 8.5% |
| v2 | 중간 증강 (aug x4) | 16,220 | 14,835 | 1,385 | 8.5% |
| v3 | 강한 증강 (aug x5) | 19,464 | 17,802 | 1,662 | 8.5% |

모든 버전에서 anomaly 비율을 ~8.5%로 일정하게 유지했습니다.
각 버전은 `data/split_versions/v{1,2,3}/` 아래의 `pretrain/` + `additional/` 디렉토리에서 수집됩니다.

## 1-2. Split 전략

- Seed: 42 (재현 가능)
- 비율: 70% train / 15% val / 15% test
- 스크립트: `scripts/make_splits_v123.py`

하나의 데이터 풀에서 용도별로 2가지 트랙의 split을 생성합니다.

### Gate 모델용 Split (이진 분류)

gate 모델은 normal vs anomaly 이진 분류이므로 양쪽 클래스 모두 포함하며,
원본 비율을 유지하는 stratified split입니다.

| Split 파일 | 용도 | 구성 |
|------------|------|------|
| `v{N}_train_gate_mix.csv` | Gate 학습 | Normal + Anomaly (70%, stratified) |
| `v{N}_val_gate.csv` | 검증 | Normal + Anomaly (15%, stratified) |
| `v{N}_test_gate.csv` | 평가 | Normal + Anomaly (15%, stratified) |

### Heatmap (PatchCore)용 Split

PatchCore는 정상 이미지만으로 학습하므로 train에는 normal만 포함합니다.

| Split 파일 | 용도 | 구성 |
|------------|------|------|
| `v{N}_train_normal.csv` | PatchCore 학습 | Normal only (70%) |
| `v{N}_val_mix.csv` | 검증 | Normal 15% + Anomaly 30% |
| `v{N}_test_mix.csv` | 평가 | Normal 15% + Anomaly 70% |

## 1-3. 버전별 상세 Split 통계

### v1 (기본 증강)

| CSV 파일 | 총 행 수 | Normal | Anomaly |
|----------|---------|--------|---------|
| v1_train_gate_mix | 9,082 | 8,307 | 775 |
| v1_val_gate | 1,946 | 1,780 | 166 |
| v1_test_gate | 1,948 | 1,781 | 167 |
| v1_train_normal | 8,307 | 8,307 | 0 |
| v1_val_mix | 2,112 | 1,780 | 332 |
| v1_test_mix | 2,557 | 1,781 | 776 |
| v1_pool (전체) | 12,976 | 11,868 | 1,108 |

### v2 (중간 증강)

| CSV 파일 | 총 행 수 | Normal | Anomaly |
|----------|---------|--------|---------|
| v2_train_gate_mix | 11,353 | 10,384 | 969 |
| v2_val_gate | 2,432 | 2,225 | 207 |
| v2_test_gate | 2,435 | 2,226 | 209 |
| v2_train_normal | 10,384 | 10,384 | 0 |
| v2_val_mix | 2,640 | 2,225 | 415 |
| v2_test_mix | 3,196 | 2,226 | 970 |
| v2_pool (전체) | 16,220 | 14,835 | 1,385 |

### v3 (강한 증강)

| CSV 파일 | 총 행 수 | Normal | Anomaly |
|----------|---------|--------|---------|
| v3_train_gate_mix | 13,624 | 12,461 | 1,163 |
| v3_val_gate | 2,919 | 2,670 | 249 |
| v3_test_gate | 2,921 | 2,671 | 250 |
| v3_train_normal | 12,461 | 12,461 | 0 |
| v3_val_mix | 3,168 | 2,670 | 498 |
| v3_test_mix | 3,835 | 2,671 | 1,164 |
| v3_pool (전체) | 19,464 | 17,802 | 1,662 |

## 1-4. 평가 전용 Split (splits_eval/)

벤치마크 비교용으로 별도 평가셋을 구성했습니다.

| CSV 파일 | 총 행 수 | Normal | Anomaly | 특징 |
|----------|---------|--------|---------|------|
| v1_balanced_test | 400 | 200 | 200 | 1:1 균형 평가 |
| v2_balanced_test | 400 | 200 | 200 | 1:1 균형 평가 |
| v3_balanced_test | 400 | 200 | 200 | 1:1 균형 평가 |
| v1_additional_test | 2,596 | 2,374 | 222 | 추가 데이터 전체 |
| v2_additional_test | 3,244 | 2,967 | 277 | 추가 데이터 전체 |
| v3_additional_test | 3,894 | 3,561 | 333 | 추가 데이터 전체 |

## 1-5. CSV 스키마

모든 CSV의 컬럼: `path, dataset_type, defect_type, label, round, split`

| 컬럼 | 설명 | 예시 |
|------|------|------|
| path | 이미지 절대 경로 | .../v1/pretrain/normal/xxx.jpg |
| dataset_type | 데이터 소스 | Kolektor |
| defect_type | 결함 유형 | surface_defect |
| label | 분류 라벨 | normal / anomaly |
| round | 버전 태그 | v1, v2, v3 |
| split | split 이름 | train_gate_mix, test_gate 등 |

---

# 2. Gate 모델 학습 설정

| 항목 | 값 |
|------|---|
| 모델 | MobileNetV3-Small (927,585 params) |
| Input size | 224 x 224 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Scheduler | Cosine Annealing |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | patience 7 (val loss 기준) |
| Backbone freeze | 처음 3 epoch |
| pos_weight | Auto (~10.7, normal/anomaly 비율로 자동 계산) |
| Calibration | Isotonic Regression |
| Seed | 42 |

---

# 3. 버전별 학습 결과

## 3-1. 학습 곡선 (Training Curves)

### v1

> 이미지 삽입: `reports/assets/v1_mnv3_small_training_curves.png`

- 24/30 epoch에서 early stopping
- train acc ~99.5%, val acc ~99.5% 수렴
- val loss가 epoch 8~10에서 spike 후 안정화

### v2

> 이미지 삽입: `reports/assets/v2_mnv3_small_training_curves.png`

- 26/30 epoch에서 early stopping
- train acc ~99.8%, val acc ~100% 도달
- train/val 모두 포화 상태

### v3

> 이미지 삽입: `reports/assets/v3_mnv3_small_training_curves.png`

- 30 epoch 전체 학습 (early stopping 미발동)
- train acc ~98.5%, val acc ~98-99%
- 가장 오래 학습했지만 v1/v2 대비 수렴이 느림

## 3-2. Test 성능 비교

| 지표 | v1 | v2 | v3 |
|------|-----|-----|-----|
| AUROC | 0.9992 | 0.9949 | 0.9969 |
| AUPRC | 0.9842 | 0.9874 | 0.9856 |
| F1 | 0.9790 | 0.9685 | 0.9778 |
| Recall | 0.9760 | 0.9569 | 0.9680 |
| Precision | 0.9819 | 0.9804 | 0.9878 |
| Optimal Threshold | 0.756 | 0.667 | 0.620 |
| Early Stop Epoch | 24 | 26 | 30 (no stop) |

## 3-3. Confusion Matrix

### v1 (test_gate: 1,948 images)

> 이미지 삽입: `reports/assets/v1_mnv3_small_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (1,781) | 1,778 | 3 |
| True Anomaly (167) | 4 | 163 |

FP: 3건, FN: 4건

### v2 (test_gate: 2,435 images)

> 이미지 삽입: `reports/assets/v2_mnv3_small_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (2,226) | 2,222 | 4 |
| True Anomaly (209) | 9 | 200 |

FP: 4건, FN: 9건

### v3 (test_gate: 2,921 images)

> 이미지 삽입: `reports/assets/v3_mnv3_small_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (2,671) | 2,668 | 3 |
| True Anomaly (250) | 8 | 242 |

FP: 3건, FN: 8건

## 3-4. ROC Curve

### v1 (AUC = 0.9992)

> 이미지 삽입: `reports/assets/v1_mnv3_small_test_roc.png`

### v2 (AUC = 0.9949)

> 이미지 삽입: `reports/assets/v2_mnv3_small_test_roc.png`

### v3 (AUC = 0.9969)

> 이미지 삽입: `reports/assets/v3_mnv3_small_test_roc.png`

## 3-5. Precision-Recall Curve

### v1 (AP = 0.9842)

> 이미지 삽입: `reports/assets/v1_mnv3_small_test_pr.png`

### v2 (AP = 0.9874)

> 이미지 삽입: `reports/assets/v2_mnv3_small_test_pr.png`

### v3 (AP = 0.9856)

> 이미지 삽입: `reports/assets/v3_mnv3_small_test_pr.png`

---

# 4. Cascade Benchmark (Baseline vs Gate-Cascade)

평가셋: `splits_eval/v{N}_balanced_test.csv` (400 images, 200:200)
Threshold: T_low=0.1, T_high=0.5

## 4-1. 성능 비교

| Version | Pipeline | Acc | F1 | Recall | Precision | Latency (ms/img) |
|---------|----------|-----|-----|--------|-----------|-------------------|
| v1 | Baseline (PatchCore only) | 0.775 | 0.796 | 0.875 | 0.729 | 71.3 |
| v1 | Gate Cascade (mnv3_small) | 0.998 | 0.998 | 0.995 | 1.000 | 33.0 |
| v2 | Baseline | 0.760 | 0.771 | 0.810 | 0.736 | 82.6 |
| v2 | Gate Cascade | 1.000 | 1.000 | 1.000 | 1.000 | 36.6 |
| v3 | Baseline | 0.653 | 0.716 | 0.875 | 0.606 | 95.1 |
| v3 | Gate Cascade | 1.000 | 1.000 | 1.000 | 1.000 | 34.4 |

## 4-2. Gate 모델 비교 (mnv3_small vs effnetb0)

| Version | Gate | Cascade F1 | Cascade Latency | Speedup |
|---------|------|-----------|----------------|---------|
| v1 | effnetb0 (4.0M params) | 0.990 | 99.8 ms | - |
| v1 | mnv3_small (928K params) | 0.998 | 33.0 ms | 3.0x |
| v2 | effnetb0 | 0.977 | 103.3 ms | - |
| v2 | mnv3_small | 1.000 | 36.6 ms | 2.8x |
| v3 | effnetb0 | 0.983 | 101.2 ms | - |
| v3 | mnv3_small | 1.000 | 34.4 ms | 2.9x |

---

# 5. 알려진 이슈: 오버피팅 우려

## 5-1. 현상

- balanced_test(200:200)에서 v2, v3 모두 F1=1.000 달성
- train/val accuracy 모두 99%+ 수렴
- 이는 오버피팅 또는 task가 너무 쉬운 것을 시사

## 5-2. 원인 분석

1. 단일 데이터 소스: v1/v2/v3 모두 KolektorSDD2 하나에서 증강한 것이라 데이터 다양성이 부족
2. 증강 복사본 겹침 가능성: 같은 원본 이미지의 다른 증강 버전이 train/test에 동시 존재 가능
3. Pretrained backbone: ImageNet pretrained MobileNetV3로 Kolektor 분류는 난이도가 매우 낮음

## 5-3. Cross-version 평가 결과 (오버피팅 검증)

v1으로만 학습한 모델을 v2, v3 테스트셋에서 평가한 결과:

| 평가셋 | F1 | Accuracy | 비고 |
|--------|-----|---------|------|
| v1_test_gate | 0.959 | 0.993 | same split |
| v1_balanced_test | 0.993 | 0.993 | same version |
| v2_test_gate | 0.943 | 0.990 | cross version |
| v2_balanced_test | 0.980 | 0.980 | cross version |
| v3_test_gate | 0.809 | 0.962 | cross version |
| v3_balanced_test | 0.965 | 0.965 | cross version |

v3 test_gate에서 F1이 0.809까지 하락합니다. 일반화 문제가 실제로 존재합니다.

## 5-4. 대응

하이퍼파라미터 서치 (`scripts/hp_search_24h.py`) 실행 중:
- Dropout 0.3 ~ 0.7
- Weight decay 1e-4 ~ 5e-2
- Label smoothing, Mixup
- Heavy augmentation
- Pretrained vs from scratch
- Backbone freeze
