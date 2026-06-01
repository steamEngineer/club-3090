# #254 — vLLM engine/registry/compose reconciliation to v0.22.0 (WORKING PLAN)

> **Status: DRAFT / not yet implemented.** Captured 2026-06-01 to revisit later.
> Delete this file in the final commit when the work lands. Tracks #254.

## Why now
Today we pruned the local docker images down to **only `vllm/vllm-openai:v0.22.0`**
(deleted `v0.21.0`/`:latest` + the purged nightlies `bf610c2f`, `01d4d1ad`,
`e47c98ef`). That fixed the *image inventory* but **not** the registry/engine layer,
which still references the deleted images. This is #254 (de-dup the literal + fix the
qwen `vllm/dual` engine↔image mismatch), now non-optional because the referenced
images no longer exist anywhere (purged from Docker Hub *and* deleted locally).

## Current state (post-prune)
**Engine profiles** (`scripts/lib/profiles/engines/`):

| engine | install.spec | state |
|---|---|---|
| `vllm-gemma-stable` | `v0.22.0` (+ vendored #40391 overlay) | ✅ alive |
| `vllm-nightly-clean` | `nightly-bf610c2f` | ❌ dead (purged+deleted) |
| `vllm-nightly-mtp` | `nightly-01d4d1ad` (Qwen Genesis v7.72.2 pin) | ❌ dead (retired) |
| `vllm-nightly-full` / `-dflash` | `nightly-e47c98ef` | ❌ dead |
| `vllm-stable` / `vllm-stable-next` | pip specs | n/a (not docker images) |

**Working composes** run `v0.22.0` only because their compose literal is
`${VLLM_IMAGE:-vllm/vllm-openai:v0.22.0}` (the `vllm-nightly-clean` engine exports an
unused `VLLM_NIGHTLY_SHA`). That literal is **silently load-bearing** — the #254 Part-1
`:?` cleanup would break them unless the engine is repointed first.

- Mis-attributed-but-working: `vllm/dual` (qwen fp8-mtp), `vllm/minimal`, `vllm/qwen-35b-a3b-dual` (fp8).
- Dead-default (default `nightly-${VLLM_NIGHTLY_SHA}` → bf610c2f, non-bootable now): qwen `bf16`,
  `tools-text`, `multi4/fp8-mtp`, `carnice`, `qwopus`, qwen-35b-a3b `preview`, gemma-4-26b-a4b ×4.
  (These are the deprecate-later set — out of scope per maintainer call.)

## Plan (one PR, dependency order)

### 1. Engine profiles
- **Add `vllm-stable-v0220`** — plain `v0.22.0`, **no overlays** (decision: do NOT rename/reuse
  `vllm-gemma-stable`, which carries the gemma #40391 overlay qwen must not mount).
  `supported_model_families`: qwen3-next-hybrid + qwen3-next-moe.
- Mark `vllm-nightly-clean` / `-full` / `-dflash` **deprecated** (purged).
- `vllm-nightly-mtp` deprecated-with-note: the retired Qwen Genesis pin; revive = re-anchor
  Genesis on a current pin (Sander cycle).

### 2. Registry (`compose_registry.py`)
- Repoint `engine=` for `vllm/dual` (fp8-mtp), `vllm/minimal`, `vllm/qwen-35b-a3b-dual` (fp8)
  → `vllm-stable-v0220`. (gemma int8/bf16-mtp stay on `vllm-gemma-stable`.)
- `DEFAULTS` already point at these — verify they resolve to `vllm-stable-v0220` after.

### 3. Production composes (#254 Part 1 — AFTER the engine repoint)
- The 5 prod composes: `image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.22.0}` →
  `image: ${VLLM_IMAGE:?set via scripts/launch.sh|switch.sh}`. Engine profile = sole version
  source; direct `docker compose` without env fails loudly instead of using a stale literal.
  - gemma `int8.yml`, `bf16-mtp.yml`  · qwen `fp8-mtp.yml`, `minimal.yml`  · qwen-35b-a3b `fp8.yml`

### 4. Scripts
- Update fixtures asserting engine↔image: `test-launch-compat`, `test-switch-registry-parity`,
  `test-launch-registry-parity`, `test-profiles-compat` (new engine + families).
- `for t in scripts/tests/*.sh` must stay green.
- *(Optional, defer)* defensive "image not pullable" warning in `launch_compat.resolve_engine_pin`
  so dead-SHA composes fail clearly.

## ⚠️ Validation gate (the real cost — do before flipping to `:?`)
Repointing assumes each qwen compose actually **runs + behaves** on *stock* v0.22.0 (no Genesis):
- `vllm/qwen-35b-a3b-dual` (fp8) — **already v0.22.0-validated** ✅.
- `vllm/dual` (27b fp8-mtp), `vllm/minimal` — **need a live boot + tool-call/MTP behavior check**
  on stock v0.22.0. If any needs a Genesis patch (P64 streaming tool-call, Cliff-2 mitigations),
  it **stays deprecated**, not repointed. Don't `:?` a compose not confirmed on the engine.

## Acceptance
- Registry engine attribution == image actually served (no "nightly-clean but runs v0.22.0").
- 5 prod composes use `${VLLM_IMAGE:?…}`; engine profile is the sole version source.
- Each repointed compose boot+behavior-validated on stock v0.22.0.
- `scripts/tests/*.sh` green. Then delete this plan file.
