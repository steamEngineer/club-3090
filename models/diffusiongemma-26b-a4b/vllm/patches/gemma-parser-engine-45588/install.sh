#!/bin/bash
# ===========================================================================
# vLLM PR #45413 + #45588 (+ #45553 gemma4_utils) — ParserEngine Gemma4 bundle
# Backport onto the pinned vllm/vllm-openai:gemma image (dgemma 74b5964f).
#
# Idempotent: no-ops if gemma4.py + engine registration are already present.
# Fails loud if expected patch files are missing from the mount.
# ===========================================================================
set -euo pipefail

VLLM=/usr/local/lib/python3.12/dist-packages/vllm
PATCHDIR=/etc/club3090/gemma-parser-engine-45588

if [ ! -d "$PATCHDIR/vllm/parser/engine" ]; then
  echo "[45588] FATAL: patch mount missing at $PATCHDIR" >&2
  exit 1
fi

if [ -f "$VLLM/parser/gemma4.py" ] \
   && grep -q 'class Gemma4Parser' "$VLLM/parser/gemma4.py" \
   && grep -q 'gemma4_engine_reasoning_parser' "$VLLM/reasoning/__init__.py" 2>/dev/null \
   && grep -q 'gemma4_tool_parser' "$VLLM/tool_parsers/__init__.py" 2>/dev/null \
   && ! grep -q 'gemma4_engine_tool_parser' "$VLLM/tool_parsers/__init__.py" 2>/dev/null; then
  echo "[45588] ParserEngine Gemma4 bundle already installed — skipping."
  exit 0
fi

echo "[45588] Installing ParserEngine Gemma4 bundle (76a373e backport)..."

mkdir -p "$VLLM/parser/engine"
cp -a "$PATCHDIR/vllm/parser/engine/." "$VLLM/parser/engine/"
cp "$PATCHDIR/vllm/parser/gemma4.py" "$VLLM/parser/gemma4.py"

cp "$PATCHDIR/vllm/reasoning/gemma4_engine_reasoning_parser.py" \
   "$VLLM/reasoning/gemma4_engine_reasoning_parser.py"
cp "$PATCHDIR/vllm/reasoning/abs_reasoning_parsers.py" \
   "$VLLM/reasoning/abs_reasoning_parsers.py"

cp "$PATCHDIR/vllm/tool_parsers/gemma4_utils.py" \
   "$VLLM/tool_parsers/gemma4_utils.py"

REASON_INIT="$VLLM/reasoning/__init__.py"
if grep -q 'gemma4_reasoning_parser' "$REASON_INIT"; then
  sed -i 's/gemma4_reasoning_parser/gemma4_engine_reasoning_parser/g' "$REASON_INIT"
  sed -i 's/Gemma4ReasoningParser/Gemma4ParserReasoningAdapter/g' "$REASON_INIT"
fi

# Tool parsing: keep LEGACY gemma4_tool_parser (bind-mounted from gemma-image-fixes).
# ParserEngine tool adapter breaks streaming extraction on DiffusionGemma block-canvas
# SSE (raw <|tool_call>…<tool_call|> leaks to content; finish_reason=stop) — see
# diffusionGemma_streaming_tool_regression.md. Revert engine tool registration if present.
TOOL_INIT="$VLLM/tool_parsers/__init__.py"
if grep -q 'gemma4_engine_tool_parser' "$TOOL_INIT"; then
  sed -i 's/gemma4_engine_tool_parser/gemma4_tool_parser/g' "$TOOL_INIT"
  sed -i 's/Gemma4EngineToolParser/Gemma4ToolParser/g' "$TOOL_INIT"
fi

echo "[45588] ParserEngine Gemma4 bundle installed (reasoning engine + legacy tool parser)."
