# 05/20 진행 내용 공유

오늘은 기존 cascade 결과에서 남아 있던 문제를 정리하는 작업을 했습니다.

핵심은 두 가지입니다.  
1. 기존에 leaky split 기준으로 남아 있던 PatchCore를 `noleak split` 기준으로 다시 학습했습니다.  
2. 그 결과를 바탕으로 `MobileNetV3-Small gate + PatchCore` 조합을 다시 벤치마크했습니다.

결론부터 말하면, **MobileNetV3-Small을 gate로 쓰는 방향은 유지해도 괜찮아 보입니다.**  
다만 모든 버전이 다 잘 나온 건 아니고, **v1은 충분히 좋았고, v2/v3는 아직 보완이 필요한 상태**입니다.

---

## 왜 다시 했는지

이전에는 Gate는 noleak으로 바꿨지만 PatchCore는 예전 split 기준으로 남아 있어서, cascade 전체 결과를 발표용으로 쓰기 애매했습니다.  
그래서 이번에는 PatchCore도 `splits_noleak/` 기준으로 다시 만들고, 그 상태에서 cascade를 다시 돌렸습니다.

---

## 이번에 한 작업

### 1. PatchCore noleak 재학습
공통 설정은 아래와 같습니다.

- Backbone: ResNet18
- Device: CPU
- Coreset ratio: 0.005
- k-NN: 9
- Train split: `v{N}_train_normal.csv`
- Eval split: `v{N}_val_mix.csv`, `v{N}_test_mix.csv`

### 2. Cascade 재벤치마크
구조는 그대로입니다.

`입력 이미지 -> MobileNetV3-Small gate -> 불확실하면 PatchCore -> 최종 판정`

---

## PatchCore 단독 결과

PatchCore만 놓고 보면 성능이 아주 강하다고 하긴 어렵습니다.

| Version | AUROC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| v1 | 0.8555 | 0.6840 | 0.7307 | 0.6429 |
| v2 | 0.7964 | 0.6220 | 0.7268 | 0.5436 |
| v3 | 0.7988 | 0.6222 | 0.7285 | 0.5429 |

그래서 이번 단계에서는 “PatchCore 단독이 좋은가?”보다,  
**“2단계 보조 판정기로서 cascade에 실제로 도움이 되는가?”**를 보는 게 더 중요하다고 판단했습니다.

---

## 최종 Cascade 결과

| Version | Accuracy | Precision | Recall | F1 | Latency / image |
|---|---:|---:|---:|---:|---:|
| v1 | 0.9954 | 0.9503 | 1.0000 | 0.9745 | 38.72 ms |
| v2 | 0.9669 | 0.7393 | 0.9628 | 0.8364 | 48.29 ms |
| v3 | 0.9700 | 0.7833 | 0.9109 | 0.8423 | 41.61 ms |

초 단위로 보면

- v1: `0.0387초/장`
- v2: `0.0483초/장`
- v3: `0.0416초/장`

Heatmap 호출률은 아래 정도였습니다.

- v1: `0.05%`
- v2: `3.72%`
- v3: `2.39%`

즉 대부분의 샘플은 gate에서 바로 처리되고, 특히 v1은 거의 gate에서 끝나는 수준이었습니다.

---

## 해석

좋았던 점은 **v1 결과가 꽤 설득력 있다는 점**입니다.  
F1이 `0.9745`, Recall이 `1.0000`이라 발표에서 대표 결과로 보여주기에도 무리가 없습니다. 속도도 이미지당 약 `0.04초` 수준으로 깔끔합니다.

반면 **v2, v3는 지금 상태 그대로 대표 결과로 세게 말하기는 어렵습니다.**  
F1이 각각 `0.8364`, `0.8423` 정도라서, 방향성은 보이지만 아직 다듬어야 할 부분이 있습니다.

---

## 지금 기준으로 정리하면

- `MobileNetV3-Small`을 gate로 쓰는 방향은 맞다.
- noleak 기준으로 다시 본 cascade는 적어도 `v1`에서는 충분히 의미 있다.
- `v2`, `v3`는 비교/분석용으로는 좋지만, 대표 성능처럼 밀기에는 아직 약하다.
- 발표에서는 **v1 중심 결과 + leakage 교훈 + 경량 gate 설계 + cascade 재검증** 흐름으로 가는 게 가장 안전하다.

---

## 발표에서 이렇게 말하면 안전할 것 같음

- data leakage를 원본 이미지 단위 split으로 해결했다
- PatchCore를 noleak 기준으로 다시 학습한 뒤 cascade를 재평가했다
- `MobileNetV3-Small gate + PatchCore` 조합은 v1에서 F1 `0.9745`, Recall `1.0000`을 기록했다
- 추론 시간은 약 `0.04초/장` 수준이었다

반대로 지금은 보수적으로 가야 하는 표현은 아래입니다.

- 모든 버전에서 cascade가 매우 우수했다
- v2/v3까지 포함해 최종 구조가 완전히 검증됐다
- PatchCore 단독 성능도 충분히 강력하다

---

## 한 줄 요약

**MobileNetV3-Small gate 방향은 유지해도 되고, noleak 기준 cascade는 v1에서 충분히 좋았습니다. 다만 v2/v3는 아직 추가 개선이 필요합니다.**

---

## 노션에 같이 붙이면 좋은 이미지

- `v1_mnv3_small_noleak_test_cm.png`
- `kfold_v123_comparison.png`
- `v1_mnv3_small_noleak_test_roc.png`
