#!/usr/bin/env python3
"""Ablation harness: E0–E3 compose mount matrix for #45588 vs PR #443."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CLUB3090_ROOT", "/root/club-3090"))
COMPOSE_DIR = REPO / "models/diffusiongemma-26b-a4b/vllm/compose/dual/fp8"
BASE = COMPOSE_DIR / "base.yml"
ENGINE = COMPOSE_DIR / "base-engine45588.yml"
PATCHES = REPO / "models/diffusiongemma-26b-a4b/vllm/patches/gemma-image-fixes"

MOUNT_LINES = {
    "reasoning": (
        "      - ../../../patches/gemma-image-fixes/gemma4_reasoning_parser.py:"
        "/usr/local/lib/python3.12/dist-packages/vllm/reasoning/gemma4_reasoning_parser.py:ro"
    ),
    "tool": (
        "      - ../../../patches/gemma-image-fixes/gemma4_tool_parser.py:"
        "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/gemma4_tool_parser.py:ro"
    ),
    "template": (
        "      - ../../../patches/gemma-image-fixes/tool_chat_template_gemma4.jinja:"
        "/vllm-workspace/examples/tool_chat_template_gemma4.jinja:ro"
    ),
}

CONFIGS = {
    "E0": {"base": "club", "engine": False, "overlays": {"reasoning", "tool", "template"}},
    "E1": {"base": "engine", "engine": True, "overlays": set()},
    "E2": {"base": "engine", "engine": True, "overlays": {"template"}},
    "E3": {"base": "engine", "engine": True, "overlays": {"tool", "template"}},
}


def build_compose(config: str) -> Path:
    spec = CONFIGS[config]
    src = BASE if spec["base"] == "club" else ENGINE
    text = src.read_text()
    if spec["overlays"]:
        extra = "\n".join(MOUNT_LINES[k] for k in ("reasoning", "tool", "template") if k in spec["overlays"])
        text = text.replace(
            "      - ../../../../../../scripts/detect_nvlink.sh:/etc/club3090/detect_nvlink.sh:ro",
            extra + "\n      - ../../../../../../scripts/detect_nvlink.sh:/etc/club3090/detect_nvlink.sh:ro",
            1,
        )
    out = COMPOSE_DIR / f".ablation-{config}.yml"
    out.write_text(text)
    return out


def restart(config: str, port: int = 8020) -> None:
    compose_file = build_compose(config)
    env = os.environ.copy()
    env["PORT"] = str(port)
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
        cwd=COMPOSE_DIR,
        check=False,
        env=env,
    )
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        cwd=COMPOSE_DIR,
        check=True,
        env=env,
    )


def wait_healthy(timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            ["curl", "-sf", "http://127.0.0.1:8020/v1/models"],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError("endpoint not healthy")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dgemma_ablation_45588.py E0|E1|E2|E3 [restart|build-only]", file=sys.stderr)
        sys.exit(1)
    config = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "restart"
    if config not in CONFIGS:
        sys.exit(f"unknown config {config}")
    path = build_compose(config)
    print(json.dumps({"config": config, "compose": str(path), "overlays": sorted(CONFIGS[config]["overlays"])}))
    if action == "build-only":
        return
    restart(config)
    wait_healthy()
    print(f"{config} ready")


if __name__ == "__main__":
    main()
