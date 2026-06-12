"""Generate a .env file from settings.DEFAULT_* constants.

Usage:
    python export_settings_env.py

This will write a `.env` file in the repository root which is used by
Docker Compose via `env_file: .env`.
"""
import os
from pathlib import Path

from settings import get_runtime_env

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ".env"


def main():
    env = get_runtime_env()
    lines = []
    for k, v in env.items():
        lines.append(f"{k}={v}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
