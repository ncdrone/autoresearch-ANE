# Multi-Agent Gossip: ANE + MLX Cross-Pollination

> Inspired by [Hyperspace's distributed autonomous ML research](https://agents.hyper.space/research-report) — 35 agents, 333 experiments, strategy compounding via P2P gossip.

**Goal:** Two autonomous agents (ANE + MLX) on the same Mac sharing experiment findings to compound faster than either alone.

**When:** After SEQ=1024 run completes (~6hrs from now). Both accelerators free to run simultaneously.

---

## Phase 1: Mono-Repo

Bring MLX training into this repo so both agents share one codebase.

**What moves in:**
- `autoresearch-mlx/train.py` → `train_mlx.py` (renamed to avoid collision with upstream)
- `autoresearch-mlx/prepare.py` → shared (both already use rustbpe vocab=8192, Karpathy data)
- `autoresearch-mlx/program.md` → `program_mlx.md`
- `autoresearch-mlx/results.tsv` → `results/mlx_results.tsv` (186 experiments, historical)
- `autoresearch-macos/results.tsv` → `results/mps_results.tsv` (84 experiments, historical)

**What stays separate:**
- `native/` — ANE-only (Obj-C, private APIs)
- `train_mac.py` — MPS training (retired but preserved)
- Dependencies: MLX needs `mlx` package, ANE is native binary

**Data pipeline:** Both already use the same data. `prepare.py` downloads + tokenizes. ANE reads `.bin` files directly (C), MLX reads via Python. No changes needed.

---

## Phase 2: Shared Results Format

Both agents log to a common format so each can read the other's findings.

**File:** `results/shared_experiments.jsonl`

```jsonl
{"ts": "2026-03-09T19:30:00", "agent": "ane", "val_bpb": 1.635, "steps": 72000, "wall_sec": 17280, "status": "keep", "config": {"lr": 2.5e-4, "seq": 512, ...}, "description": "v3b: half LR + zero-init + softcap + split LR", "lesson": "activation stability requires lower LR than short runs suggest"}
{"ts": "2026-03-09T19:35:00", "agent": "mlx", "val_bpb": 1.266, "steps": 1848, "wall_sec": 300, "status": "keep", "config": {"lr": 0.008, "seq": 1024, ...}, "description": "WARMDOWN_RATIO=0.6 + EMBEDDING_LR=1.3", "lesson": "removing softcap lets embedding LR go higher"}
```

**Fields:**
- `agent`: "ane" | "mlx" | "mps" (for historical)
- `val_bpb`: the universal metric (both compute identically)
- `config`: full hyperparameter snapshot
- `description`: what changed
- `lesson`: why it worked or didn't (the key field for gossip)

**Bootstrap:** Convert existing `results.tsv` from all three repos into this format. 325 experiments of institutional memory on day 1.

---

## Phase 3: Gossip / Inspiration Layer

Before each experiment, read peer findings and reason about what to try.

### How it works (Hyperspace's L2 + L3 pattern, adapted for 2 agents):

**L2 — Inspiration:** Agent reads last N peer experiments from `shared_experiments.jsonl`. "MLX got 1.266 by removing softcap and raising embedding LR to 1.3. Should I try that?"

**L3 — LLM Synthesis:** Agent doesn't just copy — it reasons about transferability. "MLX removed softcap because it let embedding LR go higher. ANE uses different optimizer (pure Adam vs Muon). Does the same logic apply? ANE's activation instability suggests we should be more careful — maybe raise embedding LR but keep softcap as a safety net."

### Implementation:

Add to `program.md` / `program_mlx.md`:

```
## Cross-Pollination Protocol

Before proposing your next experiment:
1. Read results/shared_experiments.jsonl (last 20 entries from peer agent)
2. Identify findings that might transfer to your framework
3. Reason about WHY the finding worked (not just WHAT changed)
4. Consider framework-specific constraints:
   - ANE: activation stability critical, longer runs, native Obj-C optimizer
   - MLX: 5-min experiments, Muon+AdamW, Python, bf16
5. Propose your experiment, noting which peer finding inspired it (if any)
6. After running, log lesson learned back to shared_experiments.jsonl
```

### What transfers vs what doesn't:

**Transfers (model-level):**
- LR ratios between param groups (the model doesn't care what hardware runs it)
- Initialization strategies (zero-init, Kaiming, Xavier)
- Architecture choices (VE, attention patterns, MLP width)
- Schedule shapes (warmdown ratio, warmup ratio)
- Regularization (weight decay, softcapping thresholds)

**Doesn't transfer (framework-level):**
- Absolute LR values (different optimizers, different batch sizes)
- Step counts (5-min vs overnight)
- Memory constraints (21GB MLX vs ANE's IOSurface model)
- Specific framework features (Muon optimizer is MLX-only)

---

## Phase 4: Run It

**Setup:**
```bash
# Terminal 1: ANE agent
cd native && ./build/train_dynamic --gossip --steps 72000 ...

# Terminal 2: MLX agent
uv run train_mlx.py  # reads program_mlx.md, runs 5-min experiments in loop
```

Both write to `results/shared_experiments.jsonl`. Both read from it before each experiment. No networking, no P2P — just a shared file on the same filesystem.

**First experiment:** Have MLX try ANE's split LR finding (matrices 0.05x). Have ANE try MLX's VE (value embeddings) or token shifting.

---

## What We Have vs Hyperspace

| | Hyperspace | Us |
|---|---|---|
| Agents | 35 | 2 |
| Hardware | 35 machines, mixed | 1 Mac, 2 accelerators |
| Communication | P2P GossipSub | Shared JSONL file |
| Experiment budget | 2-5min each | 5min (MLX), 5hr (ANE) |
| Model | 2-layer 64-dim char | 48.8M param GPT |
| Historical data | 0 (cold start) | 325 experiments (warm start) |
| Accelerator diversity | CPU vs GPU | ANE vs GPU (truly different hardware) |

Our advantage: we're not cold-starting. 325 experiments across 3 frameworks is a massive CRDT bootstrap. And our two agents use genuinely different hardware (ANE vs GPU) which forces different exploration paths — exactly what Hyperspace found produces the best compounding.

---

## Future: Jetsons

Once local gossip works, extending to the Orin 8GB is just:
1. `rsync` the shared_experiments.jsonl periodically
2. Or: simple HTTP endpoint that accepts/serves JSONL
3. Jetson runs CUDA version of train.py (original Karpathy upstream)

Three hardware targets, three frameworks, one shared knowledge base. But start local first.

---

## Success Metric

If cross-pollination works, we should see:
- ANE val_bpb improve faster than the 55-experiment solo trajectory predicted
- MLX find strategies it wouldn't have explored on its own
- At least one "second-order" discovery (combining findings from both agents into something neither had)
