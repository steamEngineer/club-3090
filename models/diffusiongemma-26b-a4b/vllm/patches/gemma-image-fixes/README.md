# DiffusionGemma — Ampere/TP + agentic fixes for the official `vllm/vllm-openai:gemma` image

Production compose ([`base.yml`](../../compose/dual/fp8/base.yml)) bind-mounts **six files**
from this directory (no #45588 boot install — rolled back 2026-06-20 after streaming
tool-call regression; see [`diffusionGemma_streaming_tool_regression.md`](./diffusionGemma_streaming_tool_regression.md)).

| File → mounts over | Fix |
|---|---|
| `marlin.py` + `marlin_utils_fp8.py` | sm_86 fp8 Marlin K-pad (warmup wall at K=352/1056) |
| `diffusion_gemma.py` | TP-vocab soft-embed + dtype fix |
| `gemma4_reasoning_parser.py` | Streaming `<\|channel>` start-token strip |
| `gemma4_tool_parser.py` | Paren recovery + quote-aware args + no-swallow |
| `tool_chat_template_gemma4.jinja` | History CoT replay strip |

The [`../gemma-parser-engine-45588/`](../gemma-parser-engine-45588/) bundle remains for
ablation only — #45588 breaks **streaming** tool extraction on block-canvas SSE.

Evidence: [`diffusionGemma_ablation_pr45588.md`](./diffusionGemma_ablation_pr45588.md),
[`diffusionGemma_streaming_tool_regression.md`](./diffusionGemma_streaming_tool_regression.md).
