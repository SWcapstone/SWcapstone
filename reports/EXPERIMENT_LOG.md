# Cascade Anomaly Detection — 실험 기록 및 현황

작성일: 2026-05-20
프로젝트: 졸업 캡스톤 — 산업용 표면 결함 검출 파이프라인
브랜치: `lightgate` (로컬) / `experiment/mnv3-small-cascade` (origin)

---

## 1. 프로젝트 목적

산업용 이미지에서 anomaly(표면 결함)를 검출하는 **2단계 cascade 파이프라인**을 설계하고 학술적으로 검증한다.

```
입력 이미지 → [Stage 1: Gate] → 정상 확신 → 통과 (빠름, 추론 비용 ↓)
                              → 불확실    → [Stage 2: PatchCore] → 정밀 판정
                              → 이상 확신 → anomaly 판정 (빠름)
```

**의도**: PatchCore(ResNet18 backbone)는 정확하지만 느리다. 가벼운 Gate 모델(MobileNetV3-Small, 2.5M params)이 대부분의 정상 이미지를 빠르게 걸러내면, PatchCore를 소수의 불확실한 이미지에만 호출하여 전체 추론 비용을 줄인다.

**학술적 기여 목표**:
1. 경량 Gate + 정밀 Judge의 cascade 구조가 단일 모델 대비 정확도와 속도를 동시에 개선함을 실증
2. Data augmentation 수준(v1/v2/v3)에 따른 성능 변화를 정량적으로 비교
3. 통계적으로 견고한 평가 방법론(5-Fold CV, Bootstrap CI)을 적용

---

## 2. 데이터셋 구성

원본 이미지 **3,244장** (normal 2,967 + anomaly 277)에 대해 3가지 증강 수준을 적용:

| Version | 증강 배수 | 총 이미지 | 설명 |
|---------|----------|----------|------|
| v1 | x4 | 12,976 | 원본 + 3가지 증강 |
| v2 | x5 | 16,220 | 원본 + 4가지 증강 |
| v3 | x6 | 19,464 | 원본 + 5가지 증강 |

**핵심**: v1/v2/v3는 서로 다른 데이터셋이 아니라, **같은 원본에 대한 증강 강도의 차이**이다.

---

## 3. 작업 타임라인

### Phase 0: 초기 파이프라인 구축 (~05-07)

기존 팀 코드(EfficientNet-B0 Gate + PatchCore)를 정리하고 v1/v2/v3 벤치마크를 수행.

- `scripts/train_gate.py` — Gate 모델 학습 (EfficientNet-B0)
- `scripts/train_patchcore.py` — PatchCore memory bank 구축
- `scripts/benchmark_pipeline.py` — Baseline(PatchCore-only) vs Cascade 비교
- `backend/src/gate_model.py` — `GateModel.load()` 키 매핑 버그 수정

**산출물**: `reports/NOTION_2026-05-07_v123_pipeline_benchmark.md`
- Cascade가 Baseline 대비 F1 +19~27%p 개선, 속도는 유사
- 하지만 이때 사용한 splits에 data leakage가 있었음 (당시 미발견)

### Phase 1: MobileNetV3-Small 경량 Gate 도입 (05-08)

**의도**: EfficientNet-B0(5.3M params) → MobileNetV3-Small(2.5M params)로 Gate를 경량화하여 cascade 1단계의 추론 속도를 개선.

- `backend/src/gate_model.py`에 `mobilenet_v3_small` backbone 추가
- `configs/noleak.yaml`에 mnv3_small 설정 추가
- v1/v2/v3 학습 → 기존 EfficientNet-B0 대비 유사 정확도, ~3배 빠른 추론

**산출물**: `reports/benchmark_light_gate_comparison.md`
- mnv3_small v1: F1=0.999 (leaky split 기준 — 의미 없음)

### Phase 2: Data Leakage 발견 및 수정 (05-19)

**발견**: 기존 `scripts/make_splits_v123.py`가 **증강 이미지 단위로 랜덤 분할**하여, 같은 원본의 서로 다른 증강 버전이 train과 test에 동시에 존재.

검증 결과:
- v1 test 이미지 중 train과 같은 원본인 비율: **98.3%**
- 사실상 답지를 보고 시험을 치는 것 → F1≈1.0은 가짜 성능

**수정**: `scripts/make_splits_v123_noleak.py` 작성
- **원본 이미지 단위로 먼저 분할** 후, 각 split 안에서만 증강을 유지
- 수정 후 train-test 원본 겹침: v1/v2/v3 모두 **0건**

**noleak split 기반 Gate 재학습** (`scripts/train_gate.py` + `configs/noleak.yaml`):

| Version | F1 | Recall | AUROC | Early Stop |
|---------|-----|--------|-------|------------|
| v1 | 0.979 | 0.976 | 0.999 | ep 10 |
| v2 | 0.922 | 0.879 | 0.982 | ep 17 |
| v3 | 0.923 | 0.884 | 0.967 | ep 11 |

**산출물**:
- `splits_noleak/` — leak-free CSV 파일들
- `models/v{1,2,3}_mnv3_small_noleak_gate.pt` — 재학습된 Gate 모델
- `models/v{1,2,3}_mnv3_small_noleak_calibrator.pkl` — Isotonic calibrator
- `models/v{1,2,3}_mnv3_small_noleak_config.json` — 학습 설정 및 메트릭
- `reports/NOTION_noleak_results.md`

### Phase 3: 통계적 검증 강화 — 5-Fold CV + Focal Loss + TTA (05-19~20)

**의도**: Phase 2의 단일 split 평가는 test anomaly가 **원본 43개**뿐이라 Bootstrap 95% CI가 넓다 (v2 Recall CI: [0.744, 0.955]). 졸업 발표에서 학술적 근거로 불충분.

**목적**:
1. 277개 anomaly 원본 전부를 평가에 활용하여 CI를 축소
2. Focal Loss로 imbalanced 데이터에서의 Recall 개선
3. TTA로 추론 안정성 확보

**방법론**:

| 기법 | 설명 | 학술 근거 |
|------|------|----------|
| Stratified 5-Fold CV | 원본 단위로 fold 분할, 277개 anomaly가 정확히 1번씩 test에 포함 | Kohavi, "A Study of Cross-Validation", IJCAI 1995 |
| Focal Loss (α=0.25, γ=2) | 쉬운 정상의 loss를 낮추고 어려운 anomaly에 집중 | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 |
| Test-Time Augmentation (5-view) | 원본 + 좌우반전 + 상하반전 + 90° + 270° 회전의 확률 평균 | 추론 시 앙상블, 추가 학습 불필요 |

**결과** (원본 단위, Bootstrap 95% CI):

| 지표 | v1 | v2 | v3 |
|------|-----|-----|-----|
| AUROC | 0.984 [0.971, 0.994] | 0.984 [0.972, 0.994] | 0.990 [0.982, 0.996] |
| AUPRC | 0.971 [0.956, 0.988] | 0.965 [0.946, 0.981] | 0.968 [0.952, 0.982] |
| F1 | **0.951** [0.932, 0.969] | 0.931 [0.908, 0.952] | 0.935 [0.912, 0.955] |
| Recall | **0.917** [0.884, 0.948] | 0.903 [0.866, 0.936] | 0.906 [0.870, 0.940] |
| Precision | 0.988 | 0.962 | 0.965 |

**핵심 해석**:
- v1(최소 증강)이 F1/Recall 최우수 — 과도한 증강은 원본 특징을 흐림
- CI가 Phase 2 대비 **~60% 축소** — "최악의 경우에도 Recall 86% 이상"
- 이전 단일 split v1 F1=0.979는 낙관적 추정이었고, 5-Fold의 0.951이 현실적 추정

**산출물**:
- `scripts/train_gate_kfold.py` — 5-Fold CV + Focal Loss + TTA 학습/평가 스크립트
- `models/v{1,2,3}_mnv3_small_kfold_focal_tta_config.json` — 전체 메트릭 + CI
- `reports/assets/kfold_v123_comparison.png` — v1/v2/v3 비교 (CI 에러바)
- `reports/assets/kfold_vs_single_comparison.png` — 단일 split vs 5-Fold 비교
- `reports/assets/kfold_data_split_overview.png` — 데이터 분할 구조 표
- `reports/assets/v{N}_mnv3_small_kfold_focal_tta_{roc,pr,cm,fold_curves}.png`

### Phase 4: Cascade 파이프라인 벤치마크 시도 (05-20) — 미완료

**의도**: Gate 단독이 아닌, Gate → PatchCore cascade 전체의 end-to-end 성능과 속도를 측정.

**시도**: `scripts/benchmark_pipeline.py`에 noleak gate 모델을 넣어 실행.

**결과와 문제점**:

| | v1 | v2 | v3 |
|---|---|---|---|
| Cascade F1 | 0.975 | 0.835 | 0.841 |
| Cascade Recall | 1.000 | 0.963 | 0.911 |
| Heatmap call rate | 0.05% | 3.72% | 2.39% |

**⚠️ 이 결과는 신뢰할 수 없다. 이유:**

1. **PatchCore가 leaky split으로 학습됨**: `models/v{1,2,3}_patchcore_r18_patchcore.pt`의 memory bank는 기존 leaky split의 train 이미지로 구축됨. test 원본의 증강 이미지가 memory bank에 포함되어 있을 가능성 높음 → v1 Recall=1.000은 PatchCore의 data leakage 오염 결과일 수 있음
2. **단일 split 평가**: Phase 3에서 Gate를 5-Fold CV로 검증했으나, cascade 벤치마크는 다시 단일 split(test anomaly 43개)으로 퇴행
3. **Baseline 비교 없음**: MPS에서 PatchCore-only baseline이 동작하지 않아 cascade만 측정

---

## 4. 현재 상태 — 완료된 것과 남은 것

### ✅ 완료

| 항목 | 상태 | 신뢰도 |
|------|------|--------|
| Data leakage 발견 및 noleak split 생성 | 완료 | 높음 |
| Gate(mnv3_small) noleak 단일 split 학습 | 완료 | 중간 (n=43) |
| Gate(mnv3_small) 5-Fold CV + Focal Loss + TTA | 완료 | **높음** (n=277, CI 보고) |
| 발표용 플롯 및 보고서 | 완료 | 높음 |

### ❌ 미완료

| 항목 | 필요 이유 | 우선순위 |
|------|----------|---------|
| **PatchCore noleak 재학습** | 현재 PatchCore memory bank에 leakage 있음. cascade 전체의 정당성 확보에 필수 | **높음** |
| **Cascade 벤치마크 (noleak 전체)** | Gate + PatchCore end-to-end 성능. 발표에서 "파이프라인 효과"를 주장하려면 필수 | **높음** |
| Cascade 5-Fold CV | cascade도 통계적으로 견고하게 평가 | 중간 |
| Baseline(PatchCore-only) vs Cascade 속도 비교 | cascade의 속도 이점 정량화 | 중간 |
| Cascade threshold(T_low, T_high) sweep on noleak | 현재 0.1/0.5는 leaky split 기준 최적값. noleak에서 재탐색 필요 | 낮음 |

---

## 5. 모델/파일 신뢰도 매핑

### Gate 모델

| 파일 | 학습 데이터 | leakage | 평가 방법 | 신뢰도 |
|------|-----------|---------|----------|--------|
| `v{N}_effnetb0_gate.pt` | leaky splits | ⚠️ 있음 | 단일 split | 사용 금지 |
| `v{N}_mnv3_small_gate.pt` | leaky splits | ⚠️ 있음 | 단일 split | 사용 금지 |
| `v{N}_mnv3_small_noleak_gate.pt` | noleak splits | ✅ 없음 | 단일 split (n=43) | 중간 |
| `v{N}_mnv3_small_kfold_focal_tta_config.json` | noleak splits | ✅ 없음 | 5-Fold CV (n=277) | **높음** |

### PatchCore 모델

| 파일 | 학습 데이터 | leakage | 신뢰도 |
|------|-----------|---------|--------|
| `v{N}_patchcore_r18_patchcore.pt` | leaky splits | ⚠️ 있음 | **사용 금지 (재학습 필요)** |

### 벤치마크 결과

| 파일 | Gate | PatchCore | 신뢰도 |
|------|------|-----------|--------|
| `benchmark_v1v2v3_balanced*.json` | leaky effnetb0 | leaky | 사용 금지 |
| `benchmark_noleak_cascade.json` | noleak mnv3_small | ⚠️ leaky | **사용 금지** |
| (미생성) noleak 전체 벤치마크 | noleak mnv3_small | noleak | 미완료 |

---

## 6. 다음 단계 (우선순위 순)

### 6-1. PatchCore noleak 재학습

```bash
python scripts/train_patchcore.py --tag v1 --heatmap patchcore_r18 \
    --splits-dir splits_noleak --device mps
# v2, v3도 동일
```

PatchCore는 `train_normal` split의 정상 이미지만으로 memory bank를 구축하므로, noleak split의 `v{N}_train_normal.csv`를 사용하면 leakage 없는 모델을 얻을 수 있다.

### 6-2. Noleak Cascade 벤치마크

noleak Gate + noleak PatchCore로 `benchmark_pipeline.py` 재실행.
Baseline(PatchCore-only) vs Cascade 비교 포함.

### 6-3. 발표 자료 최종화

현송 담당: 모델 학습 그래프 + 데이터 분할 설명
마감: 2026-05-23 금요일 20시 회의

---

## 7. 재현 방법

```bash
# 1. Noleak splits 생성
python scripts/make_splits_v123_noleak.py

# 2. Gate 모델 학습 (단일 split)
python scripts/train_gate.py --tag v1 --gate mnv3_small --config configs/noleak.yaml

# 3. Gate 5-Fold CV 평가
python scripts/train_gate_kfold.py --tag v1 --gate mnv3_small --config configs/noleak.yaml --device mps

# 4. PatchCore 학습 (TODO: noleak 재학습)
python scripts/train_patchcore.py --tag v1 --heatmap patchcore_r18 --device mps

# 5. Cascade 벤치마크
python scripts/benchmark_pipeline.py --versions v1 v2 v3 \
    --test-csv splits_noleak/v1_test_gate.csv ... \
    --gate-model models/v1_mnv3_small_noleak_gate.pt ... \
    --patchcore-model models/v1_patchcore_r18_patchcore.pt ... \
    --device cpu --output reports/benchmark_noleak_cascade.json
```

---

## 8. 핵심 교훈

1. **Data leakage는 증강 데이터에서 쉽게 발생한다.** 증강 이미지 단위가 아닌 원본 단위로 분할해야 한다.
2. **소규모 anomaly 데이터(n=277)에서는 단일 split의 point estimate를 신뢰하면 안 된다.** K-Fold CV + Bootstrap CI가 필수.
3. **파이프라인의 모든 구성 요소가 동일한 평가 기준을 충족해야 한다.** Gate만 noleak으로 고치고 PatchCore는 leaky로 두면 cascade 전체가 오염된다.
