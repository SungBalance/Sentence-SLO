# Work Log

## 2026-04-20

- Modified: Added vLLM editable install instructions to root `README.md`, root `AGENTS.md`, and `vllm/AGENTS.md`, including `sk-sslo`, `/workspace/mlsys/vllm`, git safe-directory, precompiled editable install, and build-helper prerequisites.
- Added: Documented the `HF_HOME=/cache` and `HF_HUB_CACHE=/cache/hub` cache convention in the README install/run flow.
- Debugging/verification: Confirmed `vllm` is installed editable in `sk-sslo` at `/workspace/mlsys/vllm` with version `0.0.0+sslo`, and `import vllm` succeeds.
- Modified: Updated root `AGENTS.md` and `vllm/AGENTS.md` with SSLO module placement, experiment directory, container execution, Hugging Face cache, and work-log rules.
- Added: Created root `CLAUDE.md` earlier to point Claude Code at `AGENTS.md`; created `WORKLOG.md` for ongoing session summaries.
- Debugging/verification: Checked `run_docker.sh` to confirm `sk-sslo` mounts this repo at `/workspace/mlsys` and host `/data` at container `/cache`.
- Modified: Added the rule that experiment `.sh` launch scripts should live in the same folder as their Python experiment script.
- Added: Created `exp/run_slack_bench.py` for prebuilt-vLLM slack timeline measurement without modifying or editable-installing vLLM; created `exp/run_slack_bench.sh` next to it with `HF_HOME=/cache` and `HF_HUB_CACHE=/cache/hub`.
- Debugging/verification: Confirmed the fresh `sk-sslo` container uses prebuilt vLLM `0.17.1+a03ca76a.nv26.03.46967107` with torch `2.11.0a0` CUDA `13.2`; installed the missing `datasets` benchmark dependency; ran `python3 -m compileall exp/run_slack_bench.py`; completed a smoke run on the real HF dataset `Aeala/ShareGPT_Vicuna_unfiltered` with no fixed input/output length and produced `exp/slack_results.json` plus `exp/slack_results.jsonl`.
- Modified: Added `--apply-chat-template` to `exp/run_slack_bench.py` for single-turn chat-template wrapping of sampled text prompts.
- Debugging/verification: Ran `python3 -m compileall exp/run_slack_bench.py` and completed a one-request smoke run with `--apply-chat-template`, producing `exp/slack_chat_results.json` and `exp/slack_chat_results.jsonl` with `apply_chat_template: true`.
- Modified: Updated `exp/run_slack_bench.sh` to run 1000 requests for the requested benchmark.
- Debugging/verification: Ran `exp/run_slack_bench.sh` inside `sk-sslo` using the real HF dataset `Aeala/ShareGPT_Vicuna_unfiltered`; produced `./slack_results.json` and `./slack_results.jsonl` with 1000 JSONL records, 8704 chunks, 170.69 requests/s, 80552.92 total tokens/s, and 37181.97 output tokens/s.
- Added: Created `exp/postprocess_slack.py` and `exp/postprocess_slack.sh` to compute per-chunk consume time, current deadline/slack, and cumulative deadline/slack from slack timeline JSONL.
- Debugging/verification: Ran `python3 -m compileall exp/postprocess_slack.py` and `exp/postprocess_slack.sh` inside `sk-sslo`; produced `exp/slack_chunks_postprocessed.csv` with 8704 chunk rows and the requested output columns.
- Modified: Updated slack post-processing to emit both `exp/slack_chunks_postprocessed.jsonl` and `exp/slack_chunks_postprocessed.csv`; CSV export uses pandas when available and falls back to the standard CSV writer.
- Debugging/verification: Re-ran `exp/postprocess_slack.sh` inside `sk-sslo`; verified 8704 JSONL lines and 8704 parsed CSV records.
- Modified: Merged the post-processing launch step into `exp/run_slack_bench.sh`, removed the separate `exp/postprocess_slack.sh`, and moved benchmark/postprocessed outputs under `exp/outputs/`.
- Debugging/verification: Ran `bash -n exp/run_slack_bench.sh` and `python3 -m compileall exp/run_slack_bench.py exp/postprocess_slack.py` inside `sk-sslo`.
