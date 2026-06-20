# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-only registered_adapters for club-3090 backport onto :gemma (74b5964f).

Upstream at 76a373e also registers Qwen3Parser; omitted here to avoid pulling
qwen3.py + qwen3 engine parsers that are unrelated to DiffusionGemma.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.gemma4 import Gemma4Parser

(
    Gemma4ParserReasoningAdapter,
    Gemma4ParserToolAdapter,
) = make_adapters(Gemma4Parser)
