# PR #45588 ablation — ParserEngine vs PR #443 overlays (test777, 2026-06-20)

Hermes `state.db` replays against `club3090.gridbit.internal:8020`. Pinned image
unchanged: `vllm/vllm-openai:gemma@sha256:9c719fc0…` (dgemma `74b5964f`, pre-#45588).
Engine bundle installed at boot from
[`../gemma-parser-engine-45588/`](../gemma-parser-engine-45588/) (backport of
#45413 + #45588 + #45553 `gemma4_utils.py` at `76a373e`).

Phase 1: `:gemma` tag unchanged since 2026-06-10 — no upstream republish with #45588.

## Results matrix

| Config | Engine bundle | Club reasoning | Club tool | Club template | CoT in prompt | `054214` empty (ON/OFF) | `082023` empty (ON/OFF) | `084848` tail empty | `100411` tail empty | Stream `<\|channel>` leak |
|---|---|---|---|---|---|---|---|---|---|---|
| **E0 — PR #443 baseline** ([`diffusionGemma_ablation_pr443.md`](./diffusionGemma_ablation_pr443.md) D) | no | yes | yes | yes | no | **6% / 0%** | — | 1/4 | **0/4 (4/4 tool calls)** | fixed |
| **E1 — engine only** | yes | no | no | no | yes | **6% / 0%** | **0% / 0%** | 1/4 | 2/4 | **none** (0 hits) |
| **E2 — engine + template** | yes | no | no | yes | **no** | 19% / 6% | — | 1/4 | **0/4** | — |
| **E3 — engine + template + club tool** | yes | no | yes* | yes | no | *(not run — legacy tool path dead with engine reg)* | — | — | — | — |

\*Mounting `gemma4_tool_parser.py` does not affect runtime when `__init__.py` registers
`gemma4_engine_tool_parser` — skipped.

Prior PR #443 reference (config B, parsers only, stock template): `054214` **0% / 0%**.

## Conclusions

1. **#45588 ParserEngine supersedes both legacy parser overlays** for the issues they
   fixed. E1 (engine only, stock template) matches E0/B on the canonical empty-after-tools
   replay (`054214`: **6% / 0%** vs PR #443 parsers-only **0% / 0%** — within
   diffusion stochasticity) and clears the streaming `<|channel>thought` leak (0 `channel`
   tokens in a thinking-on stream smoke test).

2. **Chat template overlay remains necessary.** #45553 sync does not stop history
   `reasoning_content` replay. E2 template render test: `cot_replayed=False`,
   `thought_blocks=0`. E1/E2 without template would replay CoT (same as stock).

3. **Deep sessions (`100411`) need engine + template.** E1 alone: **2/4** empty; E2:
   **0/4** empty (matches E0 all-six). Same pattern as PR #443 ablation (parsers +
   template additive).

4. **E2 `054214` regression (19% / 6%) vs E1 (6% / 0%)** is likely run variance — template
   does not touch tool parsing. Re-run before dropping legacy parser mounts in production.

5. **`082023` quoted-args:** E1 full replay reported `quoted-args SUMMARY: {'no_call': 10}`
   (contexts often stopped without tool calls — not directly comparable to PR #443's 7/10
   clean). No evidence engine reintroduces literal-quote JSON args on fired calls.

## Recommendation (post-#45588, pre-`:gemma` republish)

| Mount | Action |
|---|---|
| `gemma-parser-engine-45588` install.sh | **Add** (boot-time backport until `:gemma` includes #45588) |
| `gemma4_reasoning_parser.py` | **Drop** — superseded by ParserEngine |
| `gemma4_tool_parser.py` | **Drop** — superseded by ParserEngine |
| `tool_chat_template_gemma4.jinja` | **Keep** — not in #45588 |
| marlin + diffusion_gemma | **Keep** — unrelated |

Target compose: engine install + 3 Ampere/TP mounts + chat template (**4 overlays + boot
script**, down from 6 file mounts).

When vLLM republishes `:gemma` with #45588 baked in, drop the boot-time bundle and pin the
new digest.

## Artifacts

- Compose: [`base-engine45588.yml`](../../compose/dual/fp8/base-engine45588.yml)
- Harness: [`dgemma_ablation_45588.py`](../gemma-parser-engine-45588/dgemma_ablation_45588.py)
- Rig path: `/root/club-3090/…`

Production stack: [`base.yml`](../../compose/dual/fp8/base.yml) (engine install + marlin/TP + template).

Ablation-only E1 variant (no template): [`base-engine45588.yml`](../../compose/dual/fp8/base-engine45588.yml).
