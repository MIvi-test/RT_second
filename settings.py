"""Shared runtime flags from environment variables."""

from __future__ import annotations

import os

import torch

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    return default

# -- Editable defaults (change these values as your single source of truth) --
# Edit these constants and then run `python export_settings_env.py` to generate
# the .env file used by Docker Compose.
DEFAULT_USE_RERANKER: bool = True
DEFAULT_USE_GPU: bool = True
DEFAULT_USE_OLLAMA: bool = False
DEFAULT_USE_HF_CACHE: str = "/root/.cache/huggingface"
DEFAULT_STORAGE_DIR: str = "/storage"
DEFAULT_SOURCE_PATH: str = "/app/dataset_case3_v1.0_fix/"


USE_RERANKER: bool = _env_bool("USE_RERANKER", DEFAULT_USE_RERANKER)
USE_GPU: bool = _env_bool("USE_GPU", DEFAULT_USE_GPU)
USE_OLLAMA: bool = _env_bool("USE_OLLAMA", DEFAULT_USE_OLLAMA)


def get_runtime_env() -> dict:
    """Return a dict of the core environment variables suitable for writing to .env.

    This is used by the helper script `export_settings_env.py` to produce an
    env file that Docker Compose can consume.
    """
    return {
        "USE_RERANKER": str(int(DEFAULT_USE_RERANKER)),
        "USE_GPU": str(int(DEFAULT_USE_GPU)),
        "USE_OLLAMA": str(int(DEFAULT_USE_OLLAMA)),
        "HF_CACHE": DEFAULT_USE_HF_CACHE,
        "STORAGE_DIR": DEFAULT_STORAGE_DIR,
        "SOURCE_PATH": DEFAULT_SOURCE_PATH,
    }


def resolve_device() -> str:
    """Pick cuda when USE_GPU=true and CUDA is available, otherwise cpu."""
    if USE_GPU and torch.cuda.is_available():
        return "cuda"
    if USE_GPU:
        print("[WARN] USE_GPU=true, но CUDA недоступна — используем CPU")
    return "cpu"
