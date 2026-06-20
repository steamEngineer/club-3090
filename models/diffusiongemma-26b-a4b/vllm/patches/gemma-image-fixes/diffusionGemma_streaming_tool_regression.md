# DiffusionGemma — streaming `<|tool_call|>` leak regression (2026-06-20)

**Symptom:** Hermes shows raw tool protocol in assistant **content**:

```text
<|tool_call>call:web_search{query:"latest deadmau5 albums"}<tool_call|>
```

`finish_reason=stop`, `tool_calls=[]`, session `tool_call_count=0` — agent loop stalls.

**Affected sessions (Hermes `state.db`):**

| Session | When | tool_call_count | Assistant pattern |
|---|---|---|---|
| `20260620_030319_6fadd0` | latest | 0 | 1× RAW_TC_IN_CONTENT |
| `20260620_030106_c0adab` | prior | 0 | 2× RAW_TC_IN_CONTENT + 1× TEXT |

**Last good session:** `20260620_011239_e7c34c` — 11 structured `tool_calls`, ended before
engine-only migration.

## Root cause

After migrating `base.yml` to full #45588 ParserEngine (reasoning **and** tool), **streaming**
tool extraction breaks on DiffusionGemma block-canvas SSE:

| Mode | Result |
|---|---|
| `stream=true` (Hermes) | `finish_reason=stop`, raw `<|tool_call>…<tool_call|>` in **content**, 0 tool deltas |
| `stream=false` | `finish_reason=tool_calls`, structured JSON args — **works** |

Repro (100% on test777 post-migration):

```bash
python3 /tmp/dgemma_stream_tc_probe.py
# fresh_user: finish='stop' tc_deltas=0 leak=True
# preview='<|tool_call>call:web_search{query:<|"|>latest deadmau5 albums<|"|>}<tool_call|>'
```

ParserEngine receives the whole ~256-token canvas in one SSE delta; the tool state machine
does not emit `tool_calls` deltas (same failure class as the old reasoning `<|channel>` leak).

Non-streaming uses `_single_pass_parse` + `finish()` and succeeds.

## Fix (2026-06-20)

**Immediate rollback:** restore all **six** legacy bind-mounts and **disable** the #45588
boot install entirely. Verified on test777 after `--force-recreate`:

```text
fresh_user: finish='tool_calls' tc_deltas=1 leak=False
```

**Hybrid (engine reasoning + legacy tool parser) is NOT sufficient** — streaming still
leaked raw tool blocks. The engine reasoning adapter appears to break the serving-layer
streaming path for tool extraction even when the legacy tool parser file is mounted.

**Target stack until upstream fix:**

| Component | Delivery |
|---|---|
| Reasoning | legacy `gemma4_reasoning_parser.py` mount |
| Tool | legacy `gemma4_tool_parser.py` mount |
| Template | `tool_chat_template_gemma4.jinja` mount |
| Ampere/TP | marlin ×2 + `diffusion_gemma.py` |
| #45588 install | **off** |

## Model output variants seen

Hermes stored both:

- `{query:"latest deadmau5 albums"}` — plain quotes, colon separator
- `{query="latest deadmau5 albums"}` — python `=` form (also fails engine streaming)

Legacy tool parser handles both when streaming path is active.
