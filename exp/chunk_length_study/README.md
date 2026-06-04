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

## 실행

```bash
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/build_category_pools.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/run_baseline_tracking.sh'
docker exec sk-sslo-vllm bash -lc 'cd /workspace/mlsys && bash exp/chunk_length_study/analyze_chunk_length.sh'
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
