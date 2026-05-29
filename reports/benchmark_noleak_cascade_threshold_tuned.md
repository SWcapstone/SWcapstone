# Threshold-Tuned Noleak Cascade Benchmark

`T_low`, `T_high`, and PatchCore threshold were selected on `val_gate` and then fixed for `test_gate`.

| version | selected `T_low` | selected `T_high` | PatchCore threshold | accuracy | precision | recall | F1 | heatmap call rate | latency / image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0.10 | 0.50 | 0.0000 | 0.9954 | 0.9503 | 1.0000 | 0.9745 | 0.05% | 52.61 ms |
| v2 | 0.20 | 0.65 | 3.2707 | 0.9861 | 0.9738 | 0.8651 | 0.9163 | 0.53% | 50.55 ms |
| v3 | 0.20 | 0.60 | 3.1818 | 0.9867 | 0.9582 | 0.8876 | 0.9215 | 0.27% | 50.23 ms |

해석:

- `v1`은 threshold 재탐색 후에도 기존 `0.10 / 0.50`이 그대로 유지됐습니다.
- `v2`, `v3`는 noleak 기준 threshold를 다시 맞추면서 F1이 각각 `0.8364 -> 0.9163`, `0.8423 -> 0.9215`로 회복됐습니다.
- 따라서 `v2`, `v3`의 초기 저하는 전부 구조 문제라기보다 threshold 재설정이 반영되지 않았던 영향도 컸습니다.

주의:

- 이 파일은 threshold-tuned 최종 성능표입니다.
- latency는 threshold tuning 결과를 고정한 상태에서 다시 측정한 값입니다.
