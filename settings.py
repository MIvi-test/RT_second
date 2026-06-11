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


USE_RERANKER: bool = _env_bool("USE_RERANKER", False)
USE_GPU: bool = _env_bool("USE_GPU", True)


def resolve_device() -> str:
    """Pick cuda when USE_GPU=true and CUDA is available, otherwise cpu."""
    if USE_GPU and torch.cuda.is_available():
        return "cuda"
    if USE_GPU:
        print("[WARN] USE_GPU=true, но CUDA недоступна — используем CPU")
    return "cpu"
