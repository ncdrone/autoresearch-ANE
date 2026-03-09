# ANE Overnight Agent — Optimizer Upgrades + Autoresearch Loop + Long Run

You are an autonomous agent running ANE (Apple Neural Engine) training. Three phases:
1. Implement 3 optimizer improvements in `train.m` (~15 min)
2. Run autoresearch loop to find best config with new optimizer (~2 hours)
3. Launch long overnight run with winning config (~7 hours)

## Context

We're training a GPT on Apple's Neural Engine using native Obj-C with private ANE APIs. The model uses Karpathy's climbmix-400B data with rustbpe tokenizer (vocab=8192) for direct val_bpb comparison with MLX and H100.

**Current best (5-min runs):** val_bpb = 2.3878 (NL=6, SEQ=512, LR=3e-4, accum=3, warmup=25)
**MLX best:** val_bpb = 1.298 (15.7M params, SEQ=1024, 59 experiments)
**Our model:** 48.8M params, NL=6, DIM=768, SEQ=512, ~145ms/step

Previous overnight runs FAILED at step 13-17K due to activation explosions. The fixes below directly address this.

## Working Directory

```
cd /Users/dan/Dev/autoresearch-ANE/native
```

---

## PHASE 1: Implement Three Optimizer Changes (~15 min)

All changes are in `/Users/dan/Dev/autoresearch-ANE/native/training/train.m`. Make these changes ONE AT A TIME, testing each.

### Change 1: Zero-Init Output Projections (Wo and W2)

In the `from_scratch` initialization block (around line 272-288), change the init for Wo and W2 from random to zero:

**Before:**
```c
for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
...
for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
```

**After:**
```c
for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=0.0f;
...
for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=0.0f;
```

This makes the residual stream clean at initialization (each layer starts as identity). MLX does this and it dramatically improves stability.

### Change 2: Logit Softcapping

After the classifier matmul and BEFORE cross_entropy_loss (around line 536-538), add softcapping:

**Before:**
```c
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
```

**After:**
```c
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
// Logit softcapping: 15 * tanh(logits / 15) — prevents explosion
{
    float cap = 15.0f;
    float inv_cap = 1.0f / cap;
    int n_logits = CV * SEQ;
    vDSP_vsmul(logits, 1, &inv_cap, logits, 1, (vDSP_Length)n_logits);
    int nn = n_logits;
    vvtanhf(logits, logits, &nn);
    vDSP_vsmul(logits, 1, &cap, logits, 1, (vDSP_Length)n_logits);
}
float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
```

**IMPORTANT:** Also fix the backward pass. The softcapping derivative is `(1 - tanh²(x/cap))`. After `cross_entropy_loss` writes `dlogits`, we need to multiply by this derivative. Add this right after the loss computation and before `vDSP_vsmul(dlogits, 1, &loss_scale, ...)`:

```c
// Softcapping backward: dlogits *= (1 - tanh²(logits_raw / cap))
// logits now = cap*tanh(x/cap), so tanh(x/cap) = logits/cap
{
    float cap = 15.0f;
    float inv_cap = 1.0f / cap;
    int n_logits = CV * SEQ;
    float *tanh_vals = (float*)malloc(n_logits * 4);
    vDSP_vsmul(logits, 1, &inv_cap, tanh_vals, 1, (vDSP_Length)n_logits);  // tanh(x/cap) = logits/cap
    // dtanh = 1 - tanh²
    vDSP_vmul(tanh_vals, 1, tanh_vals, 1, tanh_vals, 1, (vDSP_Length)n_logits);  // tanh²
    float neg1 = -1.0f, one = 1.0f;
    vDSP_vsmsa(tanh_vals, 1, &neg1, &one, tanh_vals, 1, (vDSP_Length)n_logits);  // 1 - tanh²
    vDSP_vmul(dlogits, 1, tanh_vals, 1, dlogits, 1, (vDSP_Length)n_logits);
    free(tanh_vals);
}
```

### Change 3: Separate Learning Rates

This is the biggest expected win. MLX uses very different LRs for embeddings vs weight matrices.

Add these variables near the other hyperparameter declarations (around line 180-189):

```c
float embed_lr_scale = 2.0f;    // embeddings learn faster
float matrix_lr_scale = 0.1f;   // weight matrices learn slower (MLX's biggest win)
```

Then modify the Adam update calls (around line 961-993):

**Weight matrices — use `lr * matrix_lr_scale`:**
```c
float mlr = lr * matrix_lr_scale;
adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
```

**RMS norms — keep base `lr` (unchanged):**
```c
adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
```

**Final RMS norm — keep base `lr` (unchanged):**
```c
adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
```

**Embeddings — use `lr * embed_lr_scale`, NO weight decay:**
```c
float elr = lr * embed_lr_scale;
adam_update(embed, gembed, &aembed, adam_t, elr, adam_b1, adam_b2, adam_eps, 0.0f);
```

Note: embedding weight decay set to 0.0 (not wd). MLX does this too.

### Build and Sanity Test Each Change

After each change, rebuild and do a quick 100-step test:

```bash
cd /Users/dan/Dev/autoresearch-ANE/native
make MODEL=gpt_karpathy train
./build/train_dynamic --scratch --steps 100 --lr 3e-4 --clip 1.0 --accum 3 --warmup 25 \
  --data data/train_karpathy.bin --val data/val_karpathy.bin \
  --token-bytes data/token_bytes.bin --val-interval 50 --val-steps 5 \
  2>&1 | tail -20
```

**Check:** Compiles? Activations within ±5? Loss decreasing? No NaN/Inf? If a change breaks things, revert it and move on.

---

## PHASE 2: Autoresearch Loop (~2 hours, ~20 experiments)

Now the optimizer is different, so the old hyperparams are probably wrong. Run the autoresearch loop to find the best config.

### The Loop

1. Read current baseline val_bpb from `results/ane_karpathy_results.tsv`
2. Choose ONE hyperparameter to change
3. Run a 5-minute experiment:
   ```bash
   cd /Users/dan/Dev/autoresearch-ANE/native
   timeout 300 ./build/train_dynamic --scratch --steps <STEPS> --lr <LR> --clip <CLIP> \
     --accum <ACCUM> --warmup <WARMUP> --data data/train_karpathy.bin --val data/val_karpathy.bin \
     --token-bytes data/token_bytes.bin --val-interval 500 --val-steps 20 \
     2>&1 | tee results/run_v2_<N>.log
   ```
4. Extract val_bpb: `grep "\[VAL" results/run_v2_<N>.log | tail -1`
5. Log result to `results/ane_karpathy_results.tsv`
6. If improved: "keep" — update baseline
7. If not: "discard" — move on
8. Go to step 1

### Step Count by Config
- SEQ=512, ~145ms/step: use `--steps 1800` (fits in 5 min)
- If you change accum, step time barely changes. Keep --steps 1800.

### Sweep Order (do these in order, each builds on previous winner)

**Round 1 — Matrix LR scale (most important):**
1. `matrix_lr_scale = 0.05` (even lower matrix LR)
2. `matrix_lr_scale = 0.2` (higher)
3. `matrix_lr_scale = 0.03` (if 0.05 won)

**Round 2 — Embedding LR scale:**
4. `embed_lr_scale = 1.0` (no boost)
5. `embed_lr_scale = 5.0` (aggressive)
6. `embed_lr_scale = 3.0` (if 5.0 was too much)

**Round 3 — Base LR (with winning scales):**
7. `--lr 2e-4`
8. `--lr 5e-4`
9. `--lr 1e-3` (only if 5e-4 improved)

**Round 4 — Accumulation:**
10. `--accum 2`
11. `--accum 5`
12. `--accum 1`

**Round 5 — Warmup:**
13. `--warmup 50`
14. `--warmup 10`
15. `--warmup 100`

**Round 6 — Weight decay:**
16. `--wd 0.0`
17. `--wd 0.05`
18. `--wd 0.2`

**Round 7 — Clip:**
19. `--clip 0.5`
20. `--clip 2.0`

### How to Change matrix_lr_scale / embed_lr_scale

These are hardcoded in train.m. To change them, edit the values and recompile:
```bash
# Edit train.m to change matrix_lr_scale or embed_lr_scale
# Then rebuild:
cd /Users/dan/Dev/autoresearch-ANE/native
make MODEL=gpt_karpathy train
```

### Results Format

Log to `/Users/dan/Dev/autoresearch-ANE/results/ane_karpathy_results.tsv`:
```
run	val_bpb	config	status	description
```

Print summary after each experiment:
```
Run N: val_bpb=X.XXXX (STATUS) — description
  Best so far: X.XXXX
```

### Early Stop

If after 10+ experiments there's no improvement, skip remaining rounds and go to Phase 3 with the current best.

---

## PHASE 3: Overnight Run (~7 hours)

Take the winning config from Phase 2 and run it long.

### Calculate Steps

Measure step time from your best Phase 2 experiment. Then:
```
available_seconds = 7 * 3600 = 25200
steps = available_seconds / (step_time_ms / 1000)
```

Round DOWN to nearest 1000. This is your `--steps` value.

**CRITICAL: --steps MUST match actual expected steps.** The cosine schedule runs over --steps. If you set steps too high, the schedule barely decays and training diverges. This killed overnight v1 and v2.

### Launch

```bash
cd /Users/dan/Dev/autoresearch-ANE/native
./build/train_dynamic --scratch \
  --steps <CALCULATED_STEPS> \
  --lr <BEST_LR> \
  --clip <BEST_CLIP> \
  --accum <BEST_ACCUM> \
  --warmup <STEPS/100 or 1000, whichever is smaller> \
  --data data/train_karpathy.bin --val data/val_karpathy.bin \
  --token-bytes data/token_bytes.bin \
  --val-interval 5000 \
  --val-steps 20 \
  2>&1 | tee /Users/dan/Dev/autoresearch-ANE/results/overnight_v3_optimizer.log
```

### Monitoring

Periodically check (every 30 min or so):

```bash
# Latest val_bpb
grep "\[VAL" /Users/dan/Dev/autoresearch-ANE/results/overnight_v3_optimizer.log | tail -5

# Activation health (should stay < ±20)
grep "^step" /Users/dan/Dev/autoresearch-ANE/results/overnight_v3_optimizer.log | tail -3

# Step time consistency
grep "ms/step" /Users/dan/Dev/autoresearch-ANE/results/overnight_v3_optimizer.log | tail -3
```

**If activations exceed ±50:** The run is diverging. Kill it, reduce LR by half, recalculate remaining steps, and restart from scratch. Log the failure.

### If the Run Diverges

1. Note the step number and last good val_bpb
2. Try `--lr <current/2>` with recalculated steps for remaining time
3. If it diverges again, try `--clip 0.5`
4. If it STILL diverges, try `matrix_lr_scale = 0.03` and rebuild

---

## Key Learnings (DON'T REPEAT THESE MISTAKES)

1. **Cosine schedule --steps MUST match actual run length.** Steps=330K but running for 15K → schedule barely decays → divergence.
2. **LR=3e-4 diverges in long runs without the new optimizer fixes.** With zero-init + softcapping it should be stable. If not, drop to 2e-4.
3. **Watch x[min,max] in logs.** ±5 is healthy. ±20 is concerning. ±50 means kill it.
4. **accum=1 is surprisingly competitive** — more weight updates per time budget may beat smoother gradients.
5. **Don't change --steps mid-run.** The cosine schedule is set at launch. To change steps, restart from scratch.

## What NOT to Change

- Do NOT modify files outside `native/training/train.m`
- Do NOT change kernel code (mil_dynamic.h, ane_mil_gen.h)
- Do NOT change model config (gpt_karpathy.h) — keep NL=6, SEQ=512, DIM=768
- Do NOT change data files or conversion scripts
- Do NOT touch prepare.py, train.py, train_mac.py

## File Locations

```
~/Dev/autoresearch-ANE/native/
├── build/train_dynamic          # rebuilt binary
├── data/
│   ├── train_karpathy.bin       # 631M tokens, uint16
│   ├── val_karpathy.bin         # 63M tokens, uint16
│   └── token_bytes.bin          # int32[8192]
├── training/
│   ├── train.m                  # ← MODIFY THIS (optimizer changes)
│   ├── cpu_ops.h                # reference only
│   └── models/gpt_karpathy.h   # reference only
└── Makefile
```

Results: `~/Dev/autoresearch-ANE/results/`

## START

Begin Phase 1: Read train.m, implement the three optimizer changes one at a time with sanity tests. Then Phase 2: autoresearch loop. Then Phase 3: overnight run.
