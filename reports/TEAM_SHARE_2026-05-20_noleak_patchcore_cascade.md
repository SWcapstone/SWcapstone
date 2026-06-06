# 05/20 진행 내용 공유

브랜치는 `experiment/mnv3-small-cascade`입니다.  
오늘은 크게 두 가지를 했습니다.

첫째, 기존에 leaky split으로 학습되어 있던 PatchCore를 `splits_noleak/` 기준으로 다시 학습했습니다.  
둘째, 그렇게 다시 만든 PatchCore와 이미 정리해둔 `MobileNetV3-Small gate`를 묶어서 cascade를 다시 벤치마크했습니다.

이 작업을 다시 한 이유는 간단합니다. Gate만 noleak으로 바꿔놓고 PatchCore는 예전 split으로 두면, 전체 cascade 결과를 발표용으로 쓰기 어렵기 때문입니다. 이번에는 그 부분을 정리하는 데 집중했습니다.

---

## 1. 이번에 끝난 것

PatchCore는 `v1`, `v2`, `v3` 전부 noleak 기준으로 다시 학습했습니다. 공통 설정은 `ResNet18`, `coreset_ratio=0.005`, `k=9`, `CPU`입니다. 결과 파일은 아래 경로에 저장되어 있습니다.

- `models/v1_patchcore_r18_patchcore.pt`
- `models/v1_patchcore_r18_config.json`
- `models/v2_patchcore_r18_patchcore.pt`
- `models/v2_patchcore_r18_config.json`
- `models/v3_patchcore_r18_patchcore.pt`
- `models/v3_patchcore_r18_config.json`

실행 이력은 `reports/patchcore_run_history.jsonl`에 남기도록 해두었습니다. 같은 실험을 다시 돌려도 최소한 언제, 어떤 설정으로 돌렸는지는 추적 가능하게 만든 상태입니다.

그 다음에 아래 조합으로 cascade를 다시 측정했습니다.

```text
입력 이미지
 -> 1단계: MobileNetV3-Small gate
 -> 불확실한 경우만 PatchCore 호출
 -> 최종 normal / anomaly 판정
```

벤치마크 결과는 아래 파일에 저장됐습니다.

- `reports/benchmark_noleak_cascade_v2.json`
- `reports/benchmark_noleak_cascade_v2.md`

---

## 2. PatchCore만 다시 봤을 때

솔직히 말하면 PatchCore 단독 성능은 아주 강하다고 보긴 어렵습니다.  
이번 noleak 재학습 기준 test 성능은 대략 이 정도였습니다.

| Version | AUROC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| v1 | 0.8555 | 0.6840 | 0.7307 | 0.6429 |
| v2 | 0.7964 | 0.6220 | 0.7268 | 0.5436 |
| v3 | 0.7988 | 0.6222 | 0.7285 | 0.5429 |

그래서 이번 단계에서 중요한 질문은 “PatchCore 단독이 좋은가?”보다, “Gate 뒤에 붙는 2단계 판정기로 실제 도움이 되는가?”라고 보는 게 맞다고 생각합니다.

---

## 3. 최종 cascade 결과

이번에 다시 돌린 noleak cascade 결과는 아래와 같습니다.

| Version | Accuracy | Precision | Recall | F1 | Latency / image |
|---|---:|---:|---:|---:|---:|
| v1 | 0.9954 | 0.9503 | 1.0000 | 0.9745 | 38.72 ms |
| v2 | 0.9669 | 0.7393 | 0.9628 | 0.8364 | 48.29 ms |
| v3 | 0.9700 | 0.7833 | 0.9109 | 0.8423 | 41.61 ms |

초 단위로 바꾸면:

- v1: `0.0387초/장`
- v2: `0.0483초/장`
- v3: `0.0416초/장`

전체 test set 1회 처리 시간은 대략:

- v1: `75.7초`
- v2: `118.1초`
- v3: `122.1초`

Heatmap 호출률은 아래 정도였습니다.

- v1: `0.05%`
- v2: `3.72%`
- v3: `2.39%`

이 숫자를 보면, 실제로 대부분의 샘플은 gate에서 바로 처리되고 있습니다. 특히 v1은 거의 다 gate에서 끝난다고 봐도 될 정도입니다.

---

## 4. 제가 보는 현재 해석

좋았던 점부터 말하면, `v1`은 꽤 괜찮습니다.  
F1이 `0.9745`이고, Recall이 `1.0000`이라서 발표에서 대표 예시로 들기에도 무리가 없습니다. 속도도 이미지당 약 `0.04초` 수준이라 숫자도 깔끔합니다.

반면 `v2`, `v3`는 지금 상태 그대로 “최종 결과”처럼 세게 말하기는 어렵습니다. F1이 각각 `0.8364`, `0.8423` 수준이라, 방향성은 보이지만 대표 결과로 밀기에는 애매합니다.

결국 지금 시점에서 가장 안전한 결론은 이겁니다.

- `MobileNetV3-Small`을 gate로 쓰는 방향은 유지해도 된다.
- noleak 기준으로 다시 돌린 cascade는 적어도 `v1`에서는 충분히 의미 있다.
- 다만 `v2`, `v3`는 추가 조정 없이 바로 대표 성능처럼 말하면 과장에 가깝다.

즉 발표에서는 `v1 중심`으로 가져가고, `v2/v3`는 “증강 강도를 높인다고 항상 좋아지지 않았다”는 비교 결과로 쓰는 게 가장 자연스럽습니다.

---

## 5. 발표에서 이렇게 말하면 안전할 것 같음

써도 되는 쪽:

- data leakage를 원본 이미지 단위 split으로 해결했다
- PatchCore를 noleak 기준으로 다시 학습한 뒤 cascade를 재평가했다
- `MobileNetV3-Small gate + PatchCore` 조합은 v1에서 F1 `0.9745`, Recall `1.0000`을 기록했다
- 추론 시간은 약 `0.04초/장` 수준이었다

지금은 보수적으로 가야 하는 쪽:

- 모든 버전에서 cascade가 매우 우수했다
- v2/v3까지 포함해 최종 구조가 완전히 검증됐다
- PatchCore 단독 성능도 충분히 강력하다

---

## 6. 지금 기준으로 정리하면

모델 조합은 그대로 가면 될 것 같습니다.

- 1단계: `MobileNetV3-Small gate`
- 2단계: `PatchCore`

발표 전략은 이렇게 잡는 게 제일 무난합니다.

- 대표 결과는 `v1`
- `v2`, `v3`는 비교/분석용
- 핵심 메시지는 `leakage 교정`, `경량 gate`, `cascade 재검증`

---

## 7. 참고 파일

- PatchCore 결과: `models/v1_patchcore_r18_config.json`, `models/v2_patchcore_r18_config.json`, `models/v3_patchcore_r18_config.json`
- Cascade 결과: `reports/benchmark_noleak_cascade_v2.json`
- 간단 요약: `reports/benchmark_noleak_cascade_v2.md`
- 실행 이력: `reports/patchcore_run_history.jsonl`

---

## 한 줄로 요약하면

`MobileNetV3-Small gate` 방향은 맞고, noleak 기준 cascade는 `v1`에서 충분히 좋았습니다. 다만 `v2`, `v3`는 아직 대표 결과로 쓰기엔 조금 약합니다.
