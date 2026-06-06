# SteelVision MLOps 통합 파이프라인 기술 명세 (Visual Diagram 가이드)

본 문서는 SteelVision 프로젝트의 전체 MLOps 작동 원리를 시각화하기 위한 핵심 로직과 데이터 흐름을 상세히 설명합니다. 다이어그램 툴(Draw.io, Lucidchart, Mermaid 등)을 활용하여 그림을 그릴 때 참조하십시오.

---

## 1. 파이프라인 5대 핵심 모듈 (Building Blocks)

1.  **React Frontend (HMI/Dashboard):** 사용자가 실시간 추론 결과를 확인하고, 피드백을 주며, 재학습 및 배포를 제어하는 컨트롤 타워.
2.  **FastAPI Backend (Integration Server):** 모든 API 요청을 처리하고 PyTorch 엔진과 MinIO 저장소를 중계하는 핵심 브레인.
3.  **PyTorch Engine (Inference & Training):** 
    *   **Gate Model:** 고속 이진 분류 (Normal/Anomaly)
    *   **Heatmap Model:** PatchCore 기반 정밀 시각화
4.  **MinIO / S3 (Centralized Artifact Registry):** "진실의 원천(Source of Truth)". 모든 모델, 데이터셋, 시스템 상태를 영구 보관.
5.  **Local Storage (Hot-cache):** 실시간 성능을 위해 MinIO의 아티팩트를 임시 보관하는 컨테이너 내부 저장소.

---

## 2. 전체 데이터 루프 (The Full-Loop Story)

파이프라인은 끊임없이 순환하는 **4단계 루프**로 구성됩니다.

### Step 1. 추론 및 피드백 (Operational Phase)
*   **흐름:** 사용자 이미지 업로드 → **Gate-Heatmap Cascade 추론** → 결과 반환 (진단명 & 상세 히트맵 포함).
*   **피드백:** 모델의 오답(FP/FN) 발견 시 작업자가 UI에서 라벨 교정 → 해당 이미지는 즉시 **MinIO `feedback` 버킷**으로 전송 및 `state.json`에 기록.

### Step 2. 데이터셋 확정 (Materialization Phase)
*   **흐름:** UI에서 분류된 피드백 아이템 선택 → **[데이터셋 확정]** 버튼 클릭.
*   **로직:** 백엔드가 피드백 이미지 경로를 수집하여 **실제 학습용 CSV** 생성 → 생성된 CSV는 **MinIO `datasets` 버킷**에 버전별로 즉시 업로드.

### Step 3. 비동기 재학습 (Learning Phase)
*   **흐름:** UI에서 하이퍼파라미터(Epoch, LR 등) 설정 → **[재학습 시작]** 클릭.
*   **실제 학습:** 백엔드가 **비동기 Background Task**로 PyTorch 학습 가동.
*   **결과 생성:** 학습 완료 시 **진짜 가중치(`.pt`)**와 실험 결과가 담긴 **메타데이터(`.json`)** 쌍이 생성되어 **MinIO `models` 버킷**으로 자동 푸시.

### Step 4. 무중단 동적 배포 (Deployment Phase)
*   **흐름:** UI 모델 리스트에서 새로 학습된 모델 선택 → **[배포]** 클릭.
*   **Hot-swap:** 서버 재시작 없이 메모리 상의 모델 객체만 즉시 교체. (파일이 로컬에 없으면 MinIO에서 실시간 다운로드)
*   **자동 복구:** 서버가 꺼졌다 켜져도 MinIO의 `configs/state.json`을 통해 이전 배포 상태를 100% 복구.

---

## 3. 안정성 및 보안 아키텍처 (Safety Mechanisms)

다이어그램 구성 시 다음의 '보호 계층'을 포함하면 더 정밀한 그림이 됩니다.

*   **Deletion Protection:** "PRODUCTION" 상태인 모델 파일은 저장소에서 삭제되지 않도록 차단하는 논리 필터링 계층.
*   **Disaster Recovery Sync:** 모든 `state.json` 변경 사항이 발생할 때마다 로컬에서 MinIO로 즉시 미러링되는 동기화 라인.
*   **Discovery Layer:** 서버 실행 시 MinIO 버킷을 전수 스캔하여 UI의 모델 리스트를 자동으로 재구성하는 인벤토리 레이어.

---

## 4. 다이어그램 시각화 팁
*   **Central Hub:** MinIO를 중앙에 배치하고 모든 화살표가 거쳐가도록 그립니다.
*   **Inference Line:** (User -> API -> Models -> User) 라인은 실선으로 표시.
*   **MLOps Line:** (User -> Feedback -> Dataset -> Training -> Models) 라인은 점선 또는 다른 색상으로 표시하여 루프를 강조합니다.
*   **Artifact Pair:** 모델 파일을 그릴 때 `.pt`와 `.json`이 항상 붙어 다니는 "세트" 아이콘으로 표현하십시오.
