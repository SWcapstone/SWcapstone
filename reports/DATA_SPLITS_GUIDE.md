# 데이터 분류 및 Split 가이드

## 1. 데이터 개요

원본 데이터셋: **KolektorSDD2** (표면 결함 검출)  
증강(augmentation) 수준에 따라 3개 버전으로 구분:

| Version | 증강 수준 | 총 이미지 수 | Normal | Anomaly | Anomaly 비율 |
|---------|----------|-------------|--------|---------|-------------|
| **v1** | 기본 증강 | 12,976 | 11,868 | 1,108 | 8.5% |
| **v2** | 중간 증강 | 16,220 | 14,835 | 1,385 | 8.5% |
| **v3** | 강한 증강 | 19,464 | 17,802 | 1,662 | 8.5% |

각 버전은 `data/split_versions/v{1,2,3}/` 하위의 `pretrain/` + `additional/` 디렉토리에서 수집됩니다.

---

## 2. Split 전략

**Seed: 42** (재현 가능)  
**비율: 70% / 15% / 15%** (train / val / test)

### 두 가지 Split 트랙

하나의 데이터 풀에서 **용도별로 2가지 트랙**의 split을 생성합니다:

#### A. Heatmap (PatchCore) 용 Split

| Split | 설명 | 구성 |
|-------|------|------|
| `train_normal` | PatchCore 학습용 | Normal **only** (70%) |
| `val_mix` | 검증용 | Normal 15% + Anomaly 30% |
| `test_mix` | 평가용 | Normal 15% + Anomaly 70% |

> PatchCore는 정상 이미지만으로 학습하므로 train에는 normal만 포함.  
> val/test에는 anomaly를 섞어 이상 탐지 성능을 평가.

#### B. Gate (분류 모델) 용 Split

| Split | 설명 | 구성 |
|-------|------|------|
| `train_gate_mix` | Gate 모델 학습용 | Normal + Anomaly (70%, stratified) |
| `val_gate` | 검증용 | Normal + Anomaly (15%, stratified) |
| `test_gate` | 평가용 | Normal + Anomaly (15%, stratified) |

> Gate 모델(MobileNetV3 Small 등)은 이진 분류이므로 양쪽 클래스 모두 필요.  
> Stratified split으로 원본 비율 유지.

---

## 3. 상세 통계

### v1 (기본 증강)

| CSV 파일 | 행 수 | Normal | Anomaly |
|----------|------|--------|---------|
| `v1_pool.csv` | 12,976 | 11,868 | 1,108 |
| `v1_train_normal.csv` | 8,307 | 8,307 | 0 |
| `v1_train_gate_mix.csv` | 9,082 | 8,307 | 775 |
| `v1_val_gate.csv` | 1,946 | 1,780 | 166 |
| `v1_val_mix.csv` | 2,112 | 1,780 | 332 |
| `v1_test_gate.csv` | 1,948 | 1,781 | 167 |
| `v1_test_mix.csv` | 2,557 | 1,781 | 776 |

### v2 (중간 증강)

| CSV 파일 | 행 수 | Normal | Anomaly |
|----------|------|--------|---------|
| `v2_pool.csv` | 16,220 | 14,835 | 1,385 |
| `v2_train_normal.csv` | 10,384 | 10,384 | 0 |
| `v2_train_gate_mix.csv` | 11,353 | 10,384 | 969 |
| `v2_val_gate.csv` | 2,432 | 2,225 | 207 |
| `v2_val_mix.csv` | 2,640 | 2,225 | 415 |
| `v2_test_gate.csv` | 2,435 | 2,226 | 209 |
| `v2_test_mix.csv` | 3,196 | 2,226 | 970 |

### v3 (강한 증강)

| CSV 파일 | 행 수 | Normal | Anomaly |
|----------|------|--------|---------|
| `v3_pool.csv` | 19,464 | 17,802 | 1,662 |
| `v3_train_normal.csv` | 12,461 | 12,461 | 0 |
| `v3_train_gate_mix.csv` | 13,624 | 12,461 | 1,163 |
| `v3_val_gate.csv` | 2,919 | 2,670 | 249 |
| `v3_val_mix.csv` | 3,168 | 2,670 | 498 |
| `v3_test_gate.csv` | 2,921 | 2,671 | 250 |
| `v3_test_mix.csv` | 3,835 | 2,671 | 1,164 |

### 평가용 Split (`splits_eval/`)

| CSV 파일 | 행 수 | Normal | Anomaly | 용도 |
|----------|------|--------|---------|------|
| `v1_balanced_test.csv` | 400 | 200 | 200 | 균형 평가셋 |
| `v2_balanced_test.csv` | 400 | 200 | 200 | 균형 평가셋 |
| `v3_balanced_test.csv` | 400 | 200 | 200 | 균형 평가셋 |
| `v1_additional_test.csv` | 2,596 | 2,374 | 222 | 추가 데이터 평가 |
| `v2_additional_test.csv` | 3,244 | 2,967 | 277 | 추가 데이터 평가 |
| `v3_additional_test.csv` | 3,894 | 3,561 | 333 | 추가 데이터 평가 |

---

## 4. Round 기반 Split (멀티 데이터셋)

별도로 `round1/2/3` split도 존재합니다.  
MVTec + NEU + Kolektor 3개 데이터셋을 사용하며, 라운드별로 데이터가 누적됩니다.

| Round | Train Normal | Train Gate | Val Gate | Test Gate | Test Mix | Pool |
|-------|-------------|------------|----------|-----------|----------|------|
| round1 | 688 | 1,129 | 241 | 243 | 589 | 1,613 |
| round2 | 1,376 | 2,258 | 483 | 485 | 1,178 | 1,613 |
| round3 | 2,067 | 3,393 | 726 | 729 | 1,771 | 1,622 |

Anchor 테스트셋:
- `anchor_test_mix.csv`: 853건 (MVTec 190 + Kolektor 663)
- `anchor_neu_test.csv`: 360건 (NEU anomaly only)

---

## 5. CSV 스키마

모든 CSV 파일의 컬럼 구성:

```
path,dataset_type,defect_type,label,round,split
```

| 컬럼 | 설명 |
|------|------|
| `path` | 이미지 파일 절대 경로 |
| `dataset_type` | 데이터셋 소스 (Kolektor / MVTec / NEU) |
| `defect_type` | 결함 유형 (surface_defect, metal_nut 등) |
| `label` | `normal` 또는 `anomaly` |
| `round` | 버전/라운드 태그 (v1, v2, v3, 1, 2, 3) |
| `split` | split 이름 (train_normal, test_mix 등) |

---

## 6. 파일 위치

```
SWCapstone/
├── splits/                   # 학습/검증/테스트 split CSV
│   ├── v{1,2,3}_*.csv       # 버전별 Kolektor 증강 split
│   ├── round{1,2,3}_*.csv   # 라운드별 멀티소스 split
│   └── anchor_*.csv         # 고정 앵커 테스트셋
├── splits_eval/              # 균형 평가용 split
│   ├── v{1,2,3}_balanced_test.csv
│   └── v{1,2,3}_additional_test.csv
├── data/split_versions/      # 실제 이미지 데이터
│   ├── v1/ (pretrain + additional)
│   ├── v2/
│   └── v3/
└── scripts/
    ├── make_splits_v123.py   # v1/v2/v3 split 생성 스크립트
    └── make_splits.py        # round 기반 split 생성 스크립트
```

---

## 7. 재생성 방법

```bash
python scripts/make_splits_v123.py --versions v1 v2 v3 --out-dir splits --seed 42
```
