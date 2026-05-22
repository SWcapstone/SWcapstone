# SteelVision Backend Migration & Specification (2026-05-09)

이 문서는 SteelVision 프로젝트의 백엔드 마이그레이션 계획과 현재 구현된 핵심 기능 명세를 통합하여 기록한 문서입니다.

---

## Part 1. Backend Migration & Feature Alignment Plan (Phased)

### 1. Objective
백엔드를 기존의 레거시 "Round" 기반 데이터셋 및 모델에서 검증된 `noleak` (MobileNetV3-Small + PatchCore) 설정으로 전환합니다. 시스템 안정성을 위해 단계별로 진행하며, 백엔드 기능을 UI와 일치시키고 불필요한 레거시 코드를 제거합니다.

### 2. Key Files & Context
- **Legacy Components:** `backend/models/round*`, `splits/round*`, `backend/src/train_engine.py` (Mock logic)
- **New Components (from `experiment/` branch):** `configs/noleak.yaml`, `models/v1_*_patchcore.pt`, `models/v1_*_mnv3_small.pt`
- **Feature Gap:** `serve.py` (진단명 누락, 상세 히트맵 누락, 앙상블 상태 유지 누락)

### 3. Implementation Plan

#### Phase 1: Cleanup & Fresh Start (Preparation)
- **[데이터 정리]:** 혼선을 방지하기 위해 모든 `splits/round*` CSV와 `backend/models/round*` 가중치 파일을 삭제합니다.
- **[코드 정리]:** 
  - `data_utils.py` 및 `device_utils.py` 내의 미사용 함수 제거.
  - `gate_model.py` 및 `heatmap_model.py` 내의 방대한 레거시 주석 삭제.
  - `serve.py` 내의 중복 임포트 정리.

#### Phase 2: System Migration (Experiment Integration)
- **[설정 마이그레이션]:** `configs/config.yaml`을 `noleak.yaml` 사양(MNV3-Small 파라미터 등)에 맞게 업데이트합니다.
- **[모델 마이그레이션]:** 실험 브랜치에서 검증된 `v1` 모델들(Gate & Heatmap)을 로컬 `models/` 디렉토리로 동기화합니다.
- **[코드 업데이트]:** `dev-serve` 브랜치의 `GateModel` 클래스가 `mobilenet_v3_small` 백본과 수정된 키 리매핑 로직을 지원하도록 업데이트합니다.

#### Phase 3: UI-Backend Feature Alignment (API Update)
- **[AI 진단]:** `serve.py`에 라벨 매핑 로직을 구현하여 `/predict` 응답에 `issueType` (예: "Scratch")을 포함합니다.
- **[상세 히트맵]:** UI의 상세 보기 모드를 위해 `normalized_score_heatmap` (base64)을 추론 응답에 추가합니다.
- **[상태 유지]:** `ensemble_enabled` 설정을 `state.json`에 저장하여 서버 재시작 시에도 유지되도록 합니다.

#### Phase 4: Training Engine Revamp (Real Logic)
- **[학습 엔진]:** `train_engine.py`의 `asyncio.sleep` 기반 Mock 로직을 실제 `noleak` 분할 정책을 사용하는 `model.train_model()` 호출로 교체합니다.

### 4. Verification & Testing
- **추론 테스트:** `/predict`가 샘플 이미지에 대해 올바른 점수, 오버레이, 진단명을 반환하는지 확인합니다.
- **MLOps 테스트:** UI에서 앙상블 토글 및 모델 교체가 정상적으로 작동하고 상태가 유지되는지 확인합니다.
- **학습 테스트:** 데이터 누수 없는 실제 PyTorch 학습 프로세스가 가동되는지 짧은 학습 실행으로 확인합니다.

---

## Part 2. 현재 백엔드 상세 기능 및 구현 로직

### 1. 상태 관리 및 영속성 (State Persistence)
백엔드는 별도의 DB 없이 **JSON 파일(`state.json`)**을 데이터베이스처럼 사용하여 시스템의 전역 상태를 관리합니다.
- **구현 로직:**
    - **`_load_state()`**: 서버 시작 시 `storage/mlops/state.json` 파일을 읽어 메모리에 로드합니다. 파일이 없으면 초기 구조(기본 데이터셋 버전, 아키텍처 정보 등)를 생성합니다.
    - **`_save_state(state)`**: 데이터셋 업로드, 모델 배포, 피드백 추가 등의 변경 사항이 발생할 때마다 JSON 파일로 즉시 저장하여 **서버 재시작 후에도 상태를 유지**합니다.
    - **자동 로그 기록**: 모든 주요 작업(학습 시작, 배포 성공, 데이터 삭제 등)은 `state["logs"]`에 타임스탬프와 함께 자동 기록됩니다.

### 2. 부트스트랩 및 자동 마이그레이션 (Bootstrap Strategy)
컨테이너 실행 시 클라우드 저장소(S3/MinIO)와 로컬 환경을 동기화하고 최적의 모델을 자동으로 로드합니다.
- **구현 로직:**
    - **S3 자동 생성**: `STORAGE_TYPE="S3"` 설정 시, 실행과 동시에 필요한 버킷(`models`, `datasets`, `feedback`)이 없으면 자동으로 생성합니다.
    - **로컬 모델 S3 마이그레이션**: `startup` 이벤트 시 로컬 `models/` 폴더에 있는 `.pt` 파일들을 체크하여 S3에 없는 파일이 있다면 자동으로 업로드합니다.
    - **최신 모델 자동 로드**: S3에서 `prefix="gate"`와 `prefix="patchcore"`를 가진 가장 최근의 파일을 찾아 자동으로 추론 엔진에 올립니다. S3가 비어있으면 로컬의 최신 파일을 사용합니다.

### 3. 데이터셋 업로드 및 피드백 루프 (Data Ingestion)
UI에서 개별/대량 이미지를 업로드하고, 작업자 피드백을 통해 학습 데이터를 누적하는 기능을 제공합니다.
- **대량 업로드 (`/mlops/datasets/upload`)**:
    - UI에서 드래그 앤 드롭으로 전달된 여러 파일을 `storage/mlops/assets/uploads/`에 저장합니다.
    - 동시에 S3의 `datasets` 버킷으로 즉시 동기화합니다.
    - `state.json`의 `samples` 리스트에 최근 100개의 업로드 이력을 관리합니다.
- **작업자 피드백 (`/mlops/feedback`)**:
    - 추론 결과에 대해 사용자가 "오탐(FP)" 또는 "미탐(FN)"을 보고하면, 해당 이미지와 함께 모델의 예측 점수, 라벨을 `feedback` 버킷에 저장합니다.
    - 이는 향후 **데이터 누수 없는(noleak) 재학습 데이터셋**의 핵심 소스가 됩니다.

### 4. 무중단 모델 배포 (Hot-swap Deployment)
서버를 끄지 않고 추론 모델을 교체하는 핵심 기능입니다.
- **구현 로직 (`_perform_hot_swap`)**:
    - UI에서 특정 모델 파일(.pt)을 선택해 배포를 요청하면, 백엔드는 해당 파일이 로컬에 있는지 확인합니다.
    - 파일이 없으면 S3에서 즉시 다운로드합니다.
    - `torch.load`를 통해 메모리 상의 모델 객체 가중치만 교체합니다.
    - 이후 들어오는 `/predict` 요청은 **재시작 없이 즉시 새 모델로 처리**됩니다.

### 5. 비동기 학습 엔진 (TrainEngine)
학습 프로세스가 API 응답을 차단하지 않도록 비동기적으로 실행됩니다.
- **구현 로직**:
    - **`BackgroundTasks`**: FastAPI의 백그라운드 태스크 기능을 사용해 학습을 별도 스레드에서 돌립니다.
    - **실시간 상태 폴링**: 학습 중 발생하는 에폭(Epoch), 손실(Loss), 정확도(Acc) 지표를 전역 변수 `_training_status`에 업데이트합니다.
    - UI는 2초 간격으로 `/mlops/training/status`를 호출하여 이 지표를 가져와 실시간 차트와 프로그레스 바를 그립니다.
