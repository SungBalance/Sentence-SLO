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

- Run all project commands that execute code inside a Docker container, not on
  the host. This includes Python scripts, tests, syntax checks, benchmark runs,
  package installs, and experiment launch scripts.
- Use the `sk-sslo` Docker container by default. Use the task-specific
  container only when the experiment explicitly requires it, for example
  vLLM-Omni work may use the matching `sk-sslo-omni` container.
- Host-side shell usage should be limited to file inspection, file editing, and
  git/worktree metadata when needed. Do not use host Python, host package
  managers, or host test runners for verification.
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
- Run verification inside the appropriate Docker container. In this workspace,
  Python test dependencies may be missing; if `pytest` or `torch` is
  unavailable in the container, report that clearly and still run syntax checks
  such as `python3 -m compileall` in the container where possible.

# Work Guideline

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
