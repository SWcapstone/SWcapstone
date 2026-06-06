# Pipeline Benchmark: Baseline (PatchCore) vs Gate-Cascade

| version | n | pipeline | accuracy (mean ± std) | precision | recall | F1 | latency/img ms (mean) | latency std | total ms |
|---|---:|---|---|---|---|---|---:|---:|---:|
| v1 | 1956 | gate_cascade | 0.9954 ± 0.0000 | 0.9503 ± 0.0000 | 1.0000 ± 0.0000 | 0.9745 ± 0.0000 | 52.61 | 15.40 | 102913 |
| v2 | 2445 | gate_cascade | 0.9861 ± 0.0000 | 0.9738 ± 0.0000 | 0.8651 ± 0.0000 | 0.9163 ± 0.0000 | 50.55 | 8.91 | 123605 |
| v3 | 2934 | gate_cascade | 0.9867 ± 0.0000 | 0.9582 ± 0.0000 | 0.8876 ± 0.0000 | 0.9215 ± 0.0000 | 50.23 | 11.21 | 147385 |

> latency std는 각 run 안에서의 per-image latency 표준편차의 평균이며,
> accuracy/precision/recall/F1의 std는 run 간의 표준편차입니다.
