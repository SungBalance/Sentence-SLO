# Chunk Length Estimation 변동성 실험

per-chunk(consumable-unit) 길이 분포와 per-request 총 출력 길이 분포의 **변동성**을
비교한다. 가설: chunk 단위가 request 단위보다 변동성이 작아 다음 청크 길이 예측이
전체 출력 길이 예측보다 쉽다 (SSLO가 청크-레벨로 추정하는 설계 근거).

task(code/dialogue) × 언어(데이터 기반 top-3)로 쪼개고, prompt 길이와의 상관관계도 본다.
**baseline 정책만** 실행하며 추적한다 (baseline도 schedule_sslo를 거쳐 청크 경계 검출·기록이
동작하므로 chunks.jsonl이 생성된다).

## 데이터 현실 (관측된 제약)
- **lmsys는 gated** (컨테이너에 HF 인증 없음) → 빌더가 자동으로 **wildchat 단독**으로 폴백.
  wildchat도 다국어+코드를 포함해 두 축을 모두 커버한다.
- 언어 top-3 (8000 prompt 기준): **en / zh-cn / ru**.
- **코드 분류기(`_is_code_request`)는 영어 중심**(영어 키워드+언어명 정규식)이라 비영어 code
  표본은 거의 0 → 실질 카테고리: **en-code, en-dialogue, zh-cn-dialogue, ru-dialogue**
  (task 축은 영어 내에서, language 축은 dialogue로 비교). `MIN_POOL` 미만 풀은 자동 skip.
- 운영 주의: `run_baseline_tracking.sh`는 `run_test.sh`가 아니라 **run_test.py를 직접 호출**한다.
  길이 분포 측정엔 풀 전체가 measurement window여야 하는데 run_test.sh는
  `--warmup-target/--measurement-target`를 전달하지 않아 작은 풀에서 무한 대기하기 때문.

## 파이프라인 (3단계, 컨테이너 `sk-sslo-vllm`, `/workspace/mlsys`)

1. **카테고리 풀 생성** — `build_category_pools.sh`
   - wildchat+lmsys를 filter OFF로 로드 → `_is_code_request`(task) + `langdetect`(language)로 분류
   - 언어 top-3 자동 선택 → (lang × {code,dialogue}) 카테고리별 512 prompt 캐시 생성
   - 출력: `output/pools/<lang>-<task>/processed_dataset.jsonl`, `output/pool_meta.jsonl`,
     `output/pool_summary.json`
2. **baseline 추적 run** — `run_baseline_tracking.sh`
   - 카테고리별로 `DATASET_CACHE_DIR`를 그 풀로 가리키고 `DATASET_NAME=combine`으로
     run_test.py를 직접 호출(baseline, 9B, cap128, read), 풀 크기에 맞춘
     warmup/measurement-target로 풀 전체를 측정. (기존 스크립트 무수정)
   - 출력: `output/runs/<lang>-<task>/baseline/run_1/rate_<r>/{chunks,requests}.jsonl`
3. **분석** — `analyze_chunk_length.sh`
   - 변동성 3종: CV(=std/mean), p99/p50, IQR/median — chunk vs request 각각
   - 상관: prompt length vs (output length, mean chunk length), Pearson + Spearman
   - 출력: `output/stats/chunk_vs_request_variability.{json,csv}` +
     `output/stats/plots/{length_distribution,variability_bars,prompt_output_correlation}.png`
4. **oracle vs online posterior replay** — `replay_posterior.sh` (GPU 불필요)
   - chunks.jsonl을 `text_generation_end_time` 순으로 재생하며, 엔진과 **동일한**
     estimator(`ChunkLengthPredictor`, slo_state.py를 importlib로 직접 로드)와
     `length_tail_prob` 게이팅(warmup 128, min_denom 4, cold-start 2048)을 그대로 적용
     → ProgressServe가 각 시점에 실제로 보는 online posterior를 복원
   - **진행도별 sharpening**: progress fraction f∈{0,25,50,75%}마다 c=⌊f·L⌋ 토큰이
     나온 시점(history=그 시점까지 완료분)에서 **잔여 길이 L−c**를 조건부 median으로 예측.
     f가 커질수록 조건부 tail이 좁아져 오차가 감소 — ProgressServe가 실제로 쓰는
     "진행하며 tail이 좁아지는" 경로. (f=0은 unit 전체 길이 예측)
   - **oracle** = 해당 run 전체 분포를 다 아는 동일 predictor(사후) — oracle 오차는 분포
     자체의 변동성 floor, online−oracle 차이가 온라인 추정 비용
   - 지표: progress별 conditional-median MAE/MAPE(online vs oracle), cold 비율, oracle
     quantile 수렴 이벤트 수, tail calibration ECE(early ≤1024 vs late)
   - 출력: `output/stats/oracle_vs_posterior.{json,csv}`(long: category×progress) +
     `output/stats/plots/{posterior_convergence,posterior_sharpening,posterior_calibration}.png`
5. **consumable-unit 분포 + request divergence** — `analyze_unit_distribution.sh` (GPU 불필요)
   - 카테고리 run별 **global** consumable-unit(chunk) 길이 분포(percentile + hist).
     global = 그 run 전체 chunk pool(엔진 posterior가 run마다 reset·pool하는 단위와 동일)
   - **outlier 제외**: `--max-chunk-len`(기본 100) 초과 chunk는 문장 경계를 못 만난
     runaway 출력(consumable unit이 아님)이라 제외하고 제외 수를 보고(`0`이면 필터 끔).
     실제 문장 chunk는 p99.9 ≈ 55–82에서 끝나고 >100은 카테고리당 ≤0.08%
   - 각 request의 chunk 길이 분포가 global과 얼마나 벌어지는지 **two-sample KS**(정수 support)로
     측정 → 카테고리 무관 단일 랭킹 top-5 + 특성(median shift, CV ratio, max, prompt/output
     tokens, prompt). `MIN_CHUNKS=8` 미만 request는 제외
   - 출력: `output/stats/{unit_distribution.json, unit_distribution_global.csv,
     top_divergent_requests.csv}` +
     `output/stats/plots/{unit_distribution_global, top_divergent_requests}.png`

## 실행

```bash
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/build_category_pools.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/run_baseline_tracking.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/analyze_chunk_length.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/replay_posterior.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/analyze_unit_distribution.sh'
```

주요 env override: `LOAD_PER_DATASET`, `TARGET_PER_CATEGORY`, `TOP_LANGS`
(build); `MODEL`, `CAP`, `NUM_PROMPTS`, `RATE`, `GPU` (run).

## 재사용 (기존 코드 무수정)
- `exp/tools/lm_datasets.py`: `_load_wildchat`, `_load_lmsys`, `_is_code_request`
- `exp/tools/dataset_cache.py`: `DATASET_CACHE_DIR` env override (코드 변경 없이 풀 주입)
- `exp/run_sslo/jsonl_utils.py`: `read_jsonl`
- `exp/run_sslo/metrics_utils.py`: `percentile`, `numeric_values`
- `langdetect`: 언어 분류 (`DetectorFactory.seed=0` 고정)

## 출력 레이아웃
```
output/                       (gitignore)
  pools/<lang>-<task>/processed_dataset.jsonl
  pool_meta.jsonl, pool_summary.json
  runs/<lang>-<task>/baseline/run_1/{chunks,requests,scheduler_stats}.jsonl
  stats/chunk_vs_request_variability.{json,csv}
  stats/plots/*.png
```
