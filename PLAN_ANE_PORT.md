# Plan: Port Autoresearch GPT Architecture to ANE Training

## Problem

The ANE training code (`native/training/`) was ported from maderix/ANE which trains
LLaMA2/Qwen3 architecture. The autoresearch GPT is a different architecture. The
kernels compile and dimensions line up, but the math is wrong.

## What Works Already

- ANE runtime (compile, eval, IOSurface I/O) — proven
- Dynamic weight loading (compile-once, swap via IOSurface) — proven
- SDPA forward + 2-part backward — works (weight-free, reusable)
- Q backward and KV backward matmul kernels — works (just matmuls)
- Wo forward and Wo backward — works (just matmuls)
- RMSNorm forward/backward on CPU — works
- Embedding forward/backward on CPU — works
- Adam optimizer on CPU — works
- Gradient accumulation + clipping — works
- Async dW via GCD dispatch — works

## What Needs to Change

### 1. FFN Forward Kernel — CRITICAL

**Current** (SwiGLU, 3 weight matrices):
```
h1 = x @ W1      (DIM → HIDDEN)
h3 = x @ W3      (DIM → HIDDEN)
silu = h1 * sigmoid(h1)
gate = silu * h3
out = gate @ W2   (HIDDEN → DIM)
```

**Needed** (ReluSquared, 2 weight matrices):
```
h = x @ c_fc      (DIM → 4*DIM)
h = relu(h)^2
out = h @ c_proj   (4*DIM → DIM)
```

Changes to `mil_dynamic.h`:
- New `gen_ffn_relusq_fwd_dynamic()` generator
- Input: [1, DIM, 1, SEQ + 4*DIM + DIM] (xnorm + Wfc + Wproj)
- One matmul (x @ Wfc), then relu + square (MIL has relu op), then matmul (h @ Wproj)
- Output: [1, DIM+4*DIM, 1, SEQ] (x_next concat h_pre_act for backward)
- Residual addition stays in kernel: x_next = x2 + alpha * ffn_out

Note: HIDDEN stays 4*DIM=2048 (same size). The change is activation + 2 matrices not 3.

### 2. FFN Backward — CRITICAL

**Current**: SiLU derivative on CPU (train.m lines 560-579), then two ANE backward matmuls.

**Needed**: ReluSquared derivative is simpler:
```
d/dx [relu(x)^2] = 2 * relu(x) * (x > 0) = 2 * relu(x)
```
So: `dh = d_out * 2 * relu(h_pre_act)`  (element-wise, CPU or ANE)

Then one backward matmul instead of two:
- `dh @ c_fc^T → dx` (4*DIM → DIM) — one kernel, not two

Changes:
- Replace SiLU derivative block in train.m with: `dh[i] = dout[i] * 2.0f * fmaxf(h_pre[i], 0.0f)`
- Replace `gen_ffn_bwd_w13t_dynamic()` with single `gen_ffn_bwd_fct_dynamic()` (one matmul)
- `gen_ffn_bwd_w2t_dynamic()` becomes `gen_ffn_bwd_projt_dynamic()` (same shape, rename)
- dW computation: 2 cblas_sgemm calls instead of 3

### 3. Value Embeddings — HIGH

Alternating layers (every other layer) have a value embedding table that mixes into V:
```
ve = value_embed[token_ids]                       # [SEQ, KV_DIM]
gate = 2 * sigmoid(ve_gate(x[:, :32]))            # [SEQ, KV_HEADS]
v = v + gate.unsqueeze(-1) * ve                   # per-head gating
```

Implementation options:
- **CPU path** (simplest): After embedding lookup, compute gate on CPU, mix into V before
  packing into SDPA input IOSurface. The gate is tiny (32 → KV_HEADS matmul).
- **ANE path** (faster): Add ve mixing into the SDPA forward kernel. More complex MIL.

Recommend CPU path first. The embedding lookup and gate computation are cheap.

Changes:
- Store value embedding tables per layer (loaded from checkpoint)
- In forward loop, before SDPA: lookup ve, compute gate, add to V
- In backward loop: compute gradients for ve_gate weights and ve embeddings
- ve_gate backward: small matmul (KV_HEADS × 32), do on CPU

### 4. Learnable Residual Scalars — HIGH

Current: `res_alpha = 1/sqrt(2*NLAYERS)` (fixed scalar)
Needed: Per-layer `resid_lambdas[L]` and `x0_lambdas[L]` (learnable)

Forward:
```
x = resid_lambdas[L] * x + x0_lambdas[L] * x0
```
Where x0 is the post-embedding-norm output, carried through all layers.

Changes to train.m:
- Store x0 after initial embedding + RMSNorm
- Replace `res_alpha` with per-layer scalars in residual computation
- In forward: `x = lambda[L] * x + mu[L] * x0` before each block
- In backward: accumulate gradients for lambda[L] and mu[L]
  - `d_lambda[L] = sum(dy * x_before_scaling)`
  - `d_mu[L] = sum(dy * x0)`
  - These are scalar gradients — trivial CPU ops (dot products)
- Add lambda/mu to optimizer (Adam, separate LR group)

### 5. Softcapped Logits — HIGH

Current: raw logits → cross_entropy
Needed: `logits = 15 * tanh(logits / 15)` → cross_entropy

Changes to train.m:
- After classifier matmul (line 514-515), apply softcap on CPU:
  ```c
  for (int i = 0; i < CV*SEQ; i++)
      logits[i] = 15.0f * tanhf(logits[i] / 15.0f);
  ```
- In backward, the gradient through softcap is:
  ```c
  // d/dx [c * tanh(x/c)] = 1 - tanh(x/c)^2 = sech^2(x/c)
  for (int i = 0; i < CV*SEQ; i++) {
      float t = tanhf(raw_logits[i] / 15.0f);
      dlogits[i] *= (1.0f - t * t);
  }
  ```
- Must store raw (pre-softcap) logits for the backward pass

### 6. Post-Embedding RMSNorm — MEDIUM

Current: `embed_lookup(x_cur, embed, ...)` then straight to layers
Needed: `x = rmsnorm(embed_lookup(...))` before first layer

Changes:
- Add `rmsnorm(x_cur, x_cur, rms_embed, DIM, SEQ)` after embed_lookup in forward
- Add corresponding `rmsnorm_bwd()` at the end of the backward layer loop
- Store rms_embed weights, add to optimizer

### 7. RMSNorm Epsilon — LOW

Current: eps=1e-5 in cpu_ops.h
Needed: eps=1e-6 (PyTorch default)

One-line fix in cpu_ops.h.

### 8. Optimizer Upgrade (Muon) — LATER

Current: Adam for all params
Needed: Muon for 2D matrices, Adam for rest

This can come later. Adam will train the model correctly, just not as efficiently.
The autoresearch hyperparameters can be adapted for Adam initially.

## Kernel Compile Budget

Current (per architecture):
- sdpaFwd, woFwd, ffnFused, ffnBwdW2t, ffnBwdW13t = 5
- wotBwd, sdpaBwd1, sdpaBwd2, qBwd, kvBwd = 5
- Total: 10 kernels

After changes:
- sdpaFwd, woFwd = 2 (unchanged)
- ffnReluSqFwd = 1 (replaces ffnFused, simpler)
- ffnBwdProjT = 1 (replaces ffnBwdW2t, same shape)
- ffnBwdFcT = 1 (replaces ffnBwdW13t, simpler — one matmul not two)
- wotBwd, sdpaBwd1, sdpaBwd2, qBwd, kvBwd = 5 (unchanged)
- Total: 10 kernels (same budget!)

At 10 kernels compiled once, we're well under the ~119 limit.

## Execution Order

1. FFN forward + backward kernels (blocks training without this)
2. Softcapped logits (CPU-only change, easy)
3. Post-embedding RMSNorm (CPU-only, easy)
4. Learnable residual scalars (CPU-only, moderate)
5. Value embeddings + gates (CPU path, moderate)
6. RMSNorm epsilon fix (one line)
7. Test against train_mac.py with same weights (numerical validation)
8. Muon optimizer (optimization, not correctness)

## Files to Modify

| File | Changes |
|------|---------|
| `native/mil/mil_dynamic.h` | New FFN kernel generators (rewrite 3 functions) |
| `native/runtime/io.h` | Update spatial constants and staging functions for new FFN layout |
| `native/training/train.m` | Forward loop, backward loop, value embeds, residual scalars, softcap |
| `native/training/cpu_ops.h` | ReluSquared derivative, softcap forward/backward, epsilon fix |
| `native/training/models/gpt_autoresearch.h` | Verify constants match train_mac.py |
