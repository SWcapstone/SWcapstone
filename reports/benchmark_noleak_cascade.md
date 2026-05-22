# Pipeline Benchmark: Baseline (PatchCore) vs Gate-Cascade

| version | n | pipeline | accuracy (mean ± std) | precision | recall | F1 | latency/img ms (mean) | latency std | total ms |
|---|---:|---|---|---|---|---|---:|---:|---:|
| v1 | 1956 | gate_cascade | 0.9954 ± 0.0000 | 0.9503 ± 0.0000 | 1.0000 ± 0.0000 | 0.9745 ± 0.0000 | 64.90 | 14.95 | 126941 |
| v2 | 2445 | gate_cascade | 0.9665 ± 0.0000 | 0.7367 ± 0.0000 | 0.9628 ± 0.0000 | 0.8347 ± 0.0000 | 66.31 | 25.73 | 162125 |
| v3 | 2934 | gate_cascade | 0.9697 ± 0.0000 | 0.7807 ± 0.0000 | 0.9109 ± 0.0000 | 0.8408 ± 0.0000 | 67.59 | 30.08 | 198305 |

> latency std는 각 run 안에서의 per-image latency 표준편차의 평균이며,
> accuracy/precision/recall/F1의 std는 run 간의 표준편차입니다.
