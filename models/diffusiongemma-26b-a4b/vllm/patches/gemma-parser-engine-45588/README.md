# ParserEngine Gemma4 backport (#45413 + #45588 + #45553 utils)

Backports vLLM's engine-based Gemma4 parsers onto the pinned **`vllm/vllm-openai:gemma`**
image (dgemma branch `74b5964f`, digest `9c719fc0…`). That image predates
[#45588](https://github.com/vllm-project/vllm/pull/45588) (merged `76a373e`, 2026-06-15).

## Source commits

| PR | SHA | Files vendored |
|---|---|---|
| [#45413](https://github.com/vllm-project/vllm/pull/45413) | (in `76a373e`) | `vllm/parser/engine/*` |
| [#45588](https://github.com/vllm-project/vllm/pull/45588) | `76a373eff47a35f828636774b63ba0315e8f15d0` | `vllm/parser/gemma4.py`, engine adapters, `abs_reasoning_parsers.py`, `abstract_tool_parser.py` |
| [#45553](https://github.com/vllm-project/vllm/pull/45553) | (partial) | `vllm/tool_parsers/gemma4_utils.py` |

`registered_adapters.py` is **Gemma4-only** (no `Qwen3Parser` import) so the bundle
does not require `vllm/parser/qwen3.py`.

## Delivery

`install.sh` runs at container boot (see `base-engine45588.yml`), copies files into
site-packages, and rewrites the `gemma4` lazy-registration lines in
`reasoning/__init__.py` + `tool_parsers/__init__.py`.

**Always keep mounted separately:** `marlin.py`, `marlin_utils_fp8.py`,
`diffusion_gemma.py` (Ampere/TP — unrelated to #45588).

## Ablation configs (E0–E3)

| Config | Engine bundle | Club reasoning overlay | Club tool overlay | Club template |
|---|---|---|---|---|
| E0 (baseline) | no | yes | yes | yes |
| E1 | yes | no | no | no |
| E2 | yes | no | no | yes |
| E3 | yes | no | yes (legacy path — likely dead with engine reg) | yes |

Results: [`diffusionGemma_ablation_pr45588.md`](../gemma-image-fixes/diffusionGemma_ablation_pr45588.md).

## Retire

Drop this overlay when `:gemma` is re-pushed with #45588+ baked in, or when #45163
merges to main and club-3090 moves to a mainline pin that includes the engine parsers.
