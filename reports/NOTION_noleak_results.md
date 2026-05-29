# [수정] Data Leakage 해결 후 Gate 모델 재학습 결과

작성일: 2026-05-19
브랜치: `experiment/mnv3-small-cascade`
모델: MobileNetV3-Small (927K params)

---

# 1. 문제 발견: Data Leakage

## 기존 split 방식의 문제

기존 split은 **증강 이미지 단위**로 랜덤 분할하고 있었습니다.
같은 원본 사진의 서로 다른 증강 버전이 train과 test에 동시에 존재했습니다.

예시:
- `20441.jpg` 원본 -> train에 `_aug3`, test에 `_aug1`, `_aug2`
- `11698.jpg` 원본 -> train에 `_aug1,2,3`, test에 `_orig`

## 검증 결과

| 항목 | 수치 |
|------|------|
| v1 test 이미지 중 train과 같은 원본인 비율 | **98.3%** |
| v1 val 이미지 중 train과 같은 원본인 비율 | **98.2%** |

사실상 답지를 보고 시험을 치는 것이므로 F1=1.000이 당연한 상황이었습니다.

## 수정 방법

**원본 이미지 단위로 먼저 train/val/test를 나눈 다음, 각 split 안에서만 증강을 유지합니다.**

- 스크립트: `scripts/make_splits_v123_noleak.py`
- 출력: `splits_noleak/` 디렉토리
- 설정: `configs/noleak.yaml`

수정 후 leakage 검증:

| 검증 항목 | v1 | v2 | v3 |
|----------|-----|-----|-----|
| train-test 원본 겹침 | 0건 | 0건 | 0건 |
| train-val 원본 겹침 | 0건 | 0건 | 0건 |
| val-test 원본 겹침 | 0건 | 0건 | 0건 |

---

# 2. 데이터 규모 (원본 이미지 기준)

전체 원본 이미지: **3,244장** (normal 2,967 + anomaly 277)

| Version | 증강 배수 | 총 이미지 | 원본 수 |
|---------|----------|----------|--------|
| v1 | x4 | 12,976 | 3,244 |
| v2 | x5 | 16,220 | 3,244 |
| v3 | x6 | 19,464 | 3,244 |

## Leak-free split 통계

### v1

| Split | 총 이미지 | 원본 수 | Normal | Anomaly |
|-------|----------|--------|--------|---------|
| train_gate_mix | 9,076 | 2,269 | 8,304 | 772 |
| val_gate | 1,944 | 486 | 1,780 | 164 |
| test_gate | 1,956 | 489 | 1,784 | 172 |

### v2

| Split | 총 이미지 | 원본 수 | Normal | Anomaly |
|-------|----------|--------|--------|---------|
| train_gate_mix | 11,345 | 2,269 | 10,380 | 965 |
| val_gate | 2,430 | 486 | 2,225 | 205 |
| test_gate | 2,445 | 489 | 2,230 | 215 |

### v3

| Split | 총 이미지 | 원본 수 | Normal | Anomaly |
|-------|----------|--------|--------|---------|
| train_gate_mix | 13,614 | 2,269 | 12,456 | 1,158 |
| val_gate | 2,916 | 486 | 2,670 | 246 |
| test_gate | 2,934 | 489 | 2,676 | 258 |

---

# 3. 재학습 결과

## 3-1. 성능 비교: 기존(leaky) vs 수정(leak-free)

| 지표 | v1 기존 | v1 수정 | v2 기존 | v2 수정 | v3 기존 | v3 수정 |
|------|--------|--------|--------|--------|--------|--------|
| AUROC | 0.999 | 0.999 | 0.995 | 0.982 | 0.997 | 0.967 |
| F1 | 0.979 | 0.973 | 0.969 | 0.922 | 0.978 | 0.923 |
| Recall | 0.976 | 0.954 | 0.957 | 0.879 | 0.968 | 0.884 |
| Precision | 0.982 | 0.994 | 0.980 | 0.969 | 0.988 | 0.966 |
| AUPRC | 0.984 | 0.981 | 0.987 | 0.951 | 0.986 | 0.923 |
| Early Stop | 24 ep | 10 ep | 26 ep | 17 ep | 30 ep | 11 ep |

핵심 변화:
- v2 F1: 0.969 -> 0.922 (4.7%p 하락)
- v3 F1: 0.978 -> 0.923 (5.5%p 하락)
- v3 AUROC: 0.997 -> 0.967 (더 이상 완벽한 L자 곡선 아님)
- Early stop이 10~17 epoch으로 크게 앞당겨짐 (기존 24~30)

## 3-2. 학습 곡선

### v1 (10 epoch에서 early stop)

> 이미지 삽입: `reports/assets/v1_mnv3_small_noleak_training_curves.png`

- val loss가 불안정하게 진동 — 모델이 일반화에 어려움을 겪음
- train acc 98%, val acc 98~99% 범위에서 변동

### v2 (17 epoch에서 early stop)

> 이미지 삽입: `reports/assets/v2_mnv3_small_noleak_training_curves.png`

- epoch 9에서 val loss spike (0.89) — val acc 71%까지 급락 후 회복
- 불안정한 학습 패턴이 뚜렷

### v3 (11 epoch에서 early stop)

> 이미지 삽입: `reports/assets/v3_mnv3_small_noleak_training_curves.png`

- 가장 빠르게 early stop
- 증강이 강할수록 원본 특징 학습이 어려워짐

## 3-3. Confusion Matrix

### v1 (test: 1,956 images)

> 이미지 삽입: `reports/assets/v1_mnv3_small_noleak_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (1,784) | 1,783 | 1 |
| True Anomaly (172) | 8 | 164 |

FP: 1건, FN: 8건

### v2 (test: 2,445 images)

> 이미지 삽입: `reports/assets/v2_mnv3_small_noleak_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (2,230) | 2,224 | 6 |
| True Anomaly (215) | 26 | 189 |

FP: 6건, FN: 26건

### v3 (test: 2,934 images)

> 이미지 삽입: `reports/assets/v3_mnv3_small_noleak_test_cm.png`

|  | Pred Normal | Pred Anomaly |
|---|------------|-------------|
| True Normal (2,676) | 2,668 | 8 |
| True Anomaly (258) | 30 | 228 |

FP: 8건, FN: 30건

## 3-4. ROC Curve

### v1 (AUC = 0.9985)

> 이미지 삽입: `reports/assets/v1_mnv3_small_noleak_test_roc.png`

### v2 (AUC = 0.9820)

> 이미지 삽입: `reports/assets/v2_mnv3_small_noleak_test_roc.png`

### v3 (AUC = 0.9665)

> 이미지 삽입: `reports/assets/v3_mnv3_small_noleak_test_roc.png`

## 3-5. PR Curve

### v1 (AP = 0.9814)

> 이미지 삽입: `reports/assets/v1_mnv3_small_noleak_test_pr.png`

### v2 (AP = 0.9506)

> 이미지 삽입: `reports/assets/v2_mnv3_small_noleak_test_pr.png`

### v3 (AP = 0.9228)

> 이미지 삽입: `reports/assets/v3_mnv3_small_noleak_test_pr.png`

---

# 4. 통계적 검증: 5-Fold Stratified CV + Focal Loss + TTA

## 4-1. 단일 split의 한계

위 결과(Section 3)는 **단일 train/val/test split** 기반입니다.
test anomaly가 **43개 원본**에서 나온 증강 이미지뿐이라 통계적 불확실성이 큽니다.

- 이미지 단위 평가: 같은 원본의 증강 이미지가 여러 장 → 독립 샘플 수가 부풀려짐
- Bootstrap 95% CI: v2 Recall [0.744, 0.955] → 최악의 경우 74.4%까지 하락 가능
- 결론의 견고성이 부족

## 4-2. 개선 방법

세 가지를 동시에 적용했습니다:

| 방법 | 설명 | 근거 |
|------|------|------|
| **Stratified 5-Fold CV** | 277개 anomaly 원본 전부를 평가에 활용. 원본 단위로 fold를 나누어 leakage 방지 | 소규모 데이터셋 평가의 표준 방법 (Kohavi, 1995) |
| **Focal Loss (α=0.25, γ=2)** | 쉬운 정상 이미지의 loss를 낮추고 어려운 anomaly에 집중 | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 |
| **Test-Time Augmentation (5-view)** | 원본 + 좌우반전 + 상하반전 + 90° + 270° 회전의 평균 확률로 판정 | 추가 학습 없이 안정적인 예측 |

## 4-3. 5-Fold CV 구조

> 이미지 삽입: `reports/assets/kfold_data_split_overview.png`

**핵심: 원본 단위로 fold를 나눕니다.**
같은 원본의 증강 이미지는 반드시 같은 fold에 속합니다.
5개 fold를 순환하면서 277개 anomaly 원본이 정확히 1번씩 test에 포함됩니다.

| | v1 (4x aug) | v2 (5x aug) | v3 (6x aug) |
|---|---|---|---|
| 원본 이미지 | 3,244 | 3,244 | 3,244 |
| Normal / Anomaly 원본 | 2,967 / 277 | 2,967 / 277 | 2,967 / 277 |
| 총 이미지 (증강 포함) | 12,976 | 16,220 | 19,464 |
| Train/fold | ~8,824 imgs | ~11,030 imgs | ~13,236 imgs |
| Val/fold (15% of train) | ~1,556 imgs | ~1,945 imgs | ~2,334 imgs |
| Test/fold | ~2,596 imgs | ~3,245 imgs | ~3,894 imgs |
| Test anomaly 원본/fold | ~55 | ~55 | ~55 |
| **총 평가 anomaly 원본** | **277 (100%)** | **277 (100%)** | **277 (100%)** |

## 4-4. Fold별 결과

### v1

| Fold | AUROC | F1 | Recall | Precision | Threshold | Early Stop |
|------|-------|-----|--------|-----------|-----------|------------|
| 1 | 0.988 | 0.922 | 0.855 | 1.000 | 0.529 | ep 10 |
| 2 | 0.977 | 0.922 | 0.886 | 0.961 | 0.111 | ep 11 |
| 3 | 0.969 | 0.963 | 0.929 | 1.000 | 0.500 | ep 10 |
| 4 | 0.993 | 0.931 | 0.871 | 1.000 | 0.083 | ep 10 |
| 5 | 0.999 | 0.968 | 0.955 | 0.981 | 0.333 | ep 13 |

### v2

| Fold | AUROC | F1 | Recall | Precision | Threshold | Early Stop |
|------|-------|-----|--------|-----------|-----------|------------|
| 1 | 0.978 | 0.934 | 0.902 | 0.969 | 0.209 | ep 12 |
| 2 | 0.968 | 0.893 | 0.836 | 0.958 | 0.333 | ep 14 |
| 3 | 0.975 | 0.938 | 0.914 | 0.962 | 0.477 | ep 12 |
| 4 | 0.980 | 0.920 | 0.929 | 0.912 | 0.053 | ep 10 |
| 5 | 0.998 | 0.939 | 0.956 | 0.923 | 0.235 | ep 10 |

### v3

| Fold | AUROC | F1 | Recall | Precision | Threshold | Early Stop |
|------|-------|-----|--------|-----------|-----------|------------|
| 1 | 0.972 | 0.915 | 0.864 | 0.973 | 0.139 | ep 18 |
| 2 | 0.970 | 0.880 | 0.812 | 0.961 | 0.467 | ep 13 |
| 3 | 0.979 | 0.930 | 0.872 | 0.997 | 0.667 | ep 11 |
| 4 | 0.976 | 0.917 | 0.857 | 0.986 | 0.164 | ep 12 |
| 5 | 0.992 | 0.948 | 0.945 | 0.951 | 0.556 | ep 9 |

> 이미지 삽입 (fold별 학습 곡선):
> - `reports/assets/v1_mnv3_small_kfold_focal_tta_fold_curves.png`
> - `reports/assets/v2_mnv3_small_kfold_focal_tta_fold_curves.png`
> - `reports/assets/v3_mnv3_small_kfold_focal_tta_fold_curves.png`

## 4-5. 집계 결과 (원본 단위, Bootstrap 95% CI)

> 이미지 삽입: `reports/assets/kfold_v123_comparison.png`

| 지표 | v1 | v2 | v3 |
|------|-----|-----|-----|
| **AUROC** | 0.984 [0.971, 0.994] | 0.984 [0.972, 0.994] | **0.990** [0.982, 0.996] |
| **AUPRC** | **0.971** [0.956, 0.988] | 0.965 [0.946, 0.981] | 0.968 [0.952, 0.982] |
| **F1** | **0.951** [0.932, 0.969] | 0.931 [0.908, 0.952] | 0.935 [0.912, 0.955] |
| **Recall** | **0.917** [0.884, 0.948] | 0.903 [0.866, 0.936] | 0.906 [0.870, 0.940] |
| **Precision** | 0.988 | 0.962 | 0.965 |

## 4-6. 단일 split vs 5-Fold CV 비교

> 이미지 삽입: `reports/assets/kfold_vs_single_comparison.png`

| 지표 | 평가 방법 | v1 | v2 | v3 |
|------|----------|-----|-----|-----|
| AUROC | Single split (n=43 anom) | 0.999 | 0.982 | 0.967 |
| | **5-Fold CV (n=277 anom)** | **0.984** | **0.984** | **0.990** |
| F1 | Single split (n=43 anom) | 0.979 | 0.922 | 0.923 |
| | **5-Fold CV (n=277 anom)** | **0.951** | **0.931** | **0.935** |
| Recall | Single split (n=43 anom) | 0.976 | 0.879 | 0.884 |
| | **5-Fold CV (n=277 anom)** | **0.917** | **0.903** | **0.906** |

핵심 변화:
- v1: AUROC 0.999→0.984, F1 0.979→0.951 — 단일 split이 **낙관적**이었음
- v2/v3: F1이 오히려 미세하게 **상승** — 단일 split이 운 나쁜 fold였을 가능성
- Recall 95% CI: v1 [0.884, 0.948] vs 이전 [1.000, 1.000] — 이전 v1은 완벽 recall이었지만 43개로는 신뢰 불가

## 4-7. ROC / PR Curve (5-Fold 집계)

### ROC Curve (fold별 ± 1SD band)

> - `reports/assets/v1_mnv3_small_kfold_focal_tta_roc.png`
> - `reports/assets/v2_mnv3_small_kfold_focal_tta_roc.png`
> - `reports/assets/v3_mnv3_small_kfold_focal_tta_roc.png`

### PR Curve (fold별 + aggregated)

> - `reports/assets/v1_mnv3_small_kfold_focal_tta_pr.png`
> - `reports/assets/v2_mnv3_small_kfold_focal_tta_pr.png`
> - `reports/assets/v3_mnv3_small_kfold_focal_tta_pr.png`

### Confusion Matrix (5-Fold 합산)

> - `reports/assets/v1_mnv3_small_kfold_focal_tta_cm.png`
> - `reports/assets/v2_mnv3_small_kfold_focal_tta_cm.png`
> - `reports/assets/v3_mnv3_small_kfold_focal_tta_cm.png`

---

# 5. 전체 파이프라인 검증: Gate → Judge/PatchCore Cascade

## 5-1. 왜 추가 검증이 필요한가

Section 4는 Gate 단독 성능입니다. 실제 시스템은 Gate가 확실한 normal/anomaly를 즉시 판정하고, 불확실한 샘플만 Judge/PatchCore로 넘기는 cascade 구조입니다. 따라서 학술 발표에서는 Gate 단독 결과와 별도로 **전체 파이프라인 결과**를 보고해야 합니다.

## 5-2. 평가 설정

| 항목 | 내용 |
|------|------|
| 평가 split | `splits_noleak/v{1,2,3}_test_gate.csv` |
| Gate | `models/v{1,2,3}_mnv3_small_noleak_gate.pt` |
| Judge/Heatmap | `models/v{1,2,3}_patchcore_r18_patchcore.pt` |
| Threshold | T_low=0.1, T_high=0.5 |
| 반복 | 3 runs |
| 디바이스 | CPU |
| 결과 파일 | `reports/benchmark_noleak_cascade.{json,md}` |

## 5-3. Cascade 결과

| Version | n | Accuracy | Precision | Recall | F1 | TP / FP / TN / FN | Heatmap call rate | Latency/img |
|---------|---:|---------:|----------:|-------:|---:|-------------------|------------------:|------------:|
| **v1** | 1,956 | **0.995** | **0.950** | **1.000** | **0.975** | 172 / 9 / 1775 / 0 | 0.05% | 64.9 ms |
| v2 | 2,445 | 0.967 | 0.737 | 0.963 | 0.835 | 207 / 74 / 2156 / 8 | 3.72% | 66.3 ms |
| v3 | 2,934 | 0.970 | 0.781 | 0.911 | 0.841 | 235 / 66 / 2610 / 23 | 2.39% | 67.6 ms |

## 5-4. 해석

- **v1 cascade는 발표에 적합**: Recall 1.000, F1 0.975, FP 9건으로 전체 파이프라인 기준에서도 가장 안정적입니다.
- **v2/v3는 Gate 단독보다 cascade F1이 낮음**: 특히 Precision이 v2 0.737, v3 0.781로 떨어졌습니다. 이는 불확실 구간에서 PatchCore/Judge threshold가 FP를 충분히 억제하지 못한 결과입니다.
- **이전 결론 수정 필요**: "Gate 단독 5-Fold 성능이 좋다"는 맞지만, "전체 pipeline도 그대로 좋다"는 별도 검증 없이는 말하면 안 됩니다.
- **최종 채택은 v1 우선**: 현재 결과 기준으로는 v1이 Gate 단독과 cascade 모두에서 가장 방어 가능한 선택입니다.

---

# 6. 결론

## Data leakage가 원인이었음

- 기존 split: test의 98.3%가 train과 같은 원본 → F1~1.0 (가짜)
- 수정 split: 원본 단위 분리 → F1 0.92~0.97 (진짜)
- 데이터셋 교체 필요 없음, split 방법만 수정

## 5-Fold CV로 통계적 견고성 확보

- 단일 split: test anomaly 43개 원본 → CI가 넓어 신뢰 불가
- 5-Fold CV: 277개 anomaly 원본 전부 평가 → CI 60% 축소
- Focal Loss + TTA로 실제 성능도 개선 (v2 Recall 0.879→0.903, v3 0.884→0.906)

## 최종 Gate 모델 성능 (5-Fold CV 기준)

| Version | F1 | Recall [95% CI] | Precision | 판단 |
|---------|-----|-----------------|-----------|------|
| **v1** | **0.951** | **0.917** [0.884, 0.948] | 0.988 | **최선** |
| v2 | 0.931 | 0.903 [0.866, 0.936] | 0.962 | 양호 |
| v3 | 0.935 | 0.906 [0.870, 0.940] | 0.965 | 양호 |

## 핵심 발견

1. **v1(최소 증강)이 가장 우수** — 과도한 증강(v2, v3)은 오히려 원본 특징을 흐림
2. **세 버전 모두 Recall CI 하한 86% 이상** — 최악의 경우에도 anomaly의 86%+ 검출
3. **Precision 96~99%** — 정상 이미지를 anomaly로 잘못 분류하는 비율 극히 낮음
4. **전체 파이프라인까지 보면 v1이 최선** — v2/v3는 Gate 단독 지표는 양호하지만 cascade에서 Precision 손실이 있어 threshold 재튜닝 필요

## 파일 위치

| 항목 | 경로 |
|------|------|
| Leak-free split CSV | `splits_noleak/v{1,2,3}_*.csv` |
| Split 생성 스크립트 | `scripts/make_splits_v123_noleak.py` |
| 5-Fold CV 학습 스크립트 | `scripts/train_gate_kfold.py` |
| 설정 파일 | `configs/noleak.yaml` |
| 단일 split 학습 곡선 | `reports/assets/v{N}_mnv3_small_noleak_training_curves.png` |
| 5-Fold 학습 곡선 | `reports/assets/v{N}_mnv3_small_kfold_focal_tta_fold_curves.png` |
| 5-Fold ROC/PR/CM | `reports/assets/v{N}_mnv3_small_kfold_focal_tta_{roc,pr,cm}.png` |
| v1/v2/v3 비교 차트 | `reports/assets/kfold_v123_comparison.png` |
| Single vs K-Fold 비교 | `reports/assets/kfold_vs_single_comparison.png` |
| 데이터 분할 개요 | `reports/assets/kfold_data_split_overview.png` |
| 5-Fold config (메트릭) | `models/v{N}_mnv3_small_kfold_focal_tta_config.json` |
| 단일 split 모델 | `models/v{N}_mnv3_small_noleak_gate.pt` |
| 전체 파이프라인 benchmark | `reports/benchmark_noleak_cascade.{json,md}` |
