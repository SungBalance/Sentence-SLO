"""Stub torchaudio for Qwen3-TTS imports in the NGC vLLM container.

The container currently ships a torchaudio wheel that is ABI-incompatible with
its torch build. vLLM model inspection runs in child Python processes, so the
stub must be installed via PYTHONPATH/sitecustomize rather than only in the
parent script.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Child worker processes import this file through PYTHONPATH.
EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT))

from common.slack_utils import patch_qwen_tts_runtime

patch_qwen_tts_runtime()
