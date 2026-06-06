# Pipeline Benchmark: Baseline (PatchCore) vs Gate-Cascade

| version | n | pipeline | accuracy (mean ± std) | precision | recall | F1 | latency/img ms (mean) | latency std | total ms |
|---|---:|---|---|---|---|---|---:|---:|---:|
| v1 | 1956 | gate_cascade | 0.9954 ± 0.0000 | 0.9503 ± 0.0000 | 1.0000 ± 0.0000 | 0.9745 ± 0.0000 | 38.72 | 11.22 | 75731 |
| v2 | 2445 | gate_cascade | 0.9669 ± 0.0000 | 0.7393 ± 0.0000 | 0.9628 ± 0.0000 | 0.8364 ± 0.0000 | 48.29 | 26.43 | 118073 |
| v3 | 2934 | gate_cascade | 0.9700 ± 0.0000 | 0.7833 ± 0.0000 | 0.9109 ± 0.0000 | 0.8423 ± 0.0000 | 41.61 | 21.01 | 122069 |

> latency std는 각 run 안에서의 per-image latency 표준편차의 평균이며,
> accuracy/precision/recall/F1의 std는 run 간의 표준편차입니다.
