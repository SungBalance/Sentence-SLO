# Local Agent Instructions

These instructions apply to AI-assisted work in this workspace.

## Repository Layout

- The upstream vLLM checkout lives under `vllm/`.
- When working inside `vllm/`, also follow `vllm/AGENTS.md`.
- `vllm/CLAUDE.md` points Claude Code to the same `vllm/AGENTS.md` instructions.
- Put experiment code and experiment launch scripts under the repo-root `exp/`.
- Store experiment launch `.sh` scripts in the same folder as the Python
  experiment script they execute.

## SSLO Work

- Put new SSLO-specific vLLM package code under `vllm/vllm/sslo/`.
- Put new SSLO-specific tests under `vllm/tests/sslo/`.
- Do not recreate or import from the old `vllm.timing` module name; use `vllm.sslo`.
- If a new SSLO feature needs shared helpers, add them to `vllm/vllm/sslo/` first rather than scattering them across unrelated vLLM modules.

## Execution And Experiments

- Run commands and experiments inside the `sk-sslo` Docker container when execution is required.
- The container mounts this repo at `/workspace/mlsys/`; use that path for container-side commands.
- `run_docker.sh` mounts host `/data` to container `/cache`.
- Every experiment execution script must set the Hugging Face model cache inside the container to `/cache/`, for example with `HF_HOME=/cache` and `HF_HUB_CACHE=/cache/hub`.
- Every experiment execution `.sh` script should live next to its Python
  script, e.g. `exp/run_foo.py` and `exp/run_foo.sh`.
- Put experiment run options directly in the `.sh` script as constants. If a
  condition changes across runs, express it with shell variables and `for`
  loops inside the `.sh` file so running the script with no extra arguments is
  sufficient.

## vLLM Install In Container

- Before running vLLM experiments, install the local checkout inside `sk-sslo` from `/workspace/mlsys/vllm`.
- Mark the mounted repo as safe for git before installing: `git config --global --add safe.directory /workspace/mlsys`.
- Prefer the precompiled editable install command:
  `VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.0.0+sslo pip install -e . --no-build-isolation`.
- If metadata generation reports missing build helpers, install them first:
  `pip install "setuptools_scm>=8.0" "setuptools>=77,<81" "packaging>=24.2" wheel ninja "cmake>=3.26.1" jinja2`.
- Avoid plain isolated `pip install -e .` in this container unless needed; it may spend a long time reinstalling large build dependencies.

## Work Log

- After each work session, append a concise note to repo-root `WORKLOG.md`.
- Include three categories when applicable: modified content, added content, and debugging/verification details.

## Verification

- Prefer targeted checks for the files changed.
- In this workspace, Python test dependencies may be missing; if `pytest` or `torch` is unavailable, report that clearly and still run syntax checks such as `python3 -m compileall` where possible.
