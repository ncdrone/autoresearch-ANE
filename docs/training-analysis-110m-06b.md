# Training Analysis Report: 110M & 0.6B Models
## maderix/ANE (1024-dim) vs Karpathy (768-dim) Architectures

**Date:** 2026-03-16
**Purpose:** Establish training cost baselines for comparison against Rustane

---

## 1. Architecture Configurations

### Karpathy Architecture (768-dim)

Based on `gpt_karpathy.h` — ReluSquared FFN (non-gated), MHA, no GQA.

| Config | 110M | 0.6B |
|--------|------|------|
| DIM | 768 | 1536 |
| HIDDEN | 2048 | 4096 |
| HEADS | 6 | 12 |
| KV_HEADS | 6 (MHA) | 12 (MHA) |
| HEAD_DIM | 128 | 128 |
| NLAYERS | 19 | 27 |
| VOCAB | 8192 | 8192 |
| SEQ | 512 (ANE) / 2048 (GPU) | 2048 (GPU only) |
| **Params** | **~111M** | **~607M** |

**Parameter breakdown (768-dim, per layer):**
- Attention: 4 x 768^2 = 2,359,296
- FFN (ReluSquared, non-gated): 2 x 768 x 2048 = 3,145,728
- **Per layer total: 5,505,024**

**Parameter breakdown (1536-dim, per layer):**
- Attention: 4 x 1536^2 = 9,437,184
- FFN: 2 x 1536 x 4096 = 12,582,912
- **Per layer total: 22,020,096**

**110M total:** 19 layers x 5.5M + 8192 x 768 embed = 104.6M + 6.3M = **110.9M**
**0.6B total:** 27 layers x 22.0M + 8192 x 1536 embed = 594.5M + 12.6M = **607.1M**

### maderix/ANE Architecture (1024-dim)

Based on `qwen3_06b.h` — SiLU gated FFN, GQA (2:1 ratio), explicit HEAD_DIM=128.

| Config | 110M | 0.6B |
|--------|------|------|
| DIM | 1024 | 1024 |
| HIDDEN | 3072 | 3072 |
| HEADS | 16 | 16 |
| KV_HEADS | 8 (GQA 2:1) | 8 (GQA 2:1) |
| HEAD_DIM | 128 | 128 |
| Q_DIM | 2048 | 2048 |
| KV_DIM | 1024 | 1024 |
| NLAYERS | 7 | 28 |
| VOCAB | 8192 | 8192 |
| SEQ | 256 (ANE) / 2048 (GPU) | 2048 (GPU only) |
| **Params** | **~115M** | **~580M** |

**Parameter breakdown (per layer):**
- Wq: 1024 x 2048 = 2,097,152
- Wk: 1024 x 1024 = 1,048,576
- Wv: 1024 x 1024 = 1,048,576
- Wo: 2048 x 1024 = 2,097,152
- W1 (gate): 1024 x 3072 = 3,145,728
- W3 (up): 1024 x 3072 = 3,145,728
- W2 (down): 3072 x 1024 = 3,145,728
- **Per layer total: 15,728,640**

**110M total:** 7 layers x 15.7M + 8192 x 1024 embed = 110.1M + 8.4M = **118.5M** (or 6 layers = **102.8M**)
**0.6B total:** 28 layers x 15.7M + 8192 x 1024 embed = 440.4M + 8.4M = **448.8M**

> **Note:** The full Qwen3-0.6B uses vocab=151,936 which accounts for ~155M params in the embedding alone,
> pushing it to ~596M total. With rustbpe vocab=8192, the 28-layer config is only ~449M.
> To hit true 0.6B at vocab=8192, you'd need **~37 layers** (37 x 15.7M + 8.4M = **590M**).

---

## 2. Compute Requirements (FLOPs)

Using the standard approximation: **6 x N x D** total training FLOPs (forward + backward),
where N = parameters, D = tokens trained on.

### Chinchilla-Optimal Token Counts

| Model Size | Optimal Tokens | Source |
|-----------|---------------|--------|
| 110M | ~2.2B | 20x params (Chinchilla) |
| 0.6B | ~12B | 20x params (Chinchilla) |

### Total Training FLOPs

| Model | Tokens | Total FLOPs |
|-------|--------|-------------|
| 110M (either arch) | 2.2B | **1.45 x 10^18** |
| 0.6B (either arch) | 12B | **4.32 x 10^19** |

> Architecture (768 vs 1024 dim) doesn't change total FLOPs significantly at the same param count.
> The difference is in per-step efficiency and hardware utilization.

---

## 3. Hardware Throughput (Measured from This Repo)

| Hardware | Sustained FLOP/s | Source |
|----------|-------------------|--------|
| **H100 (CUDA)** | ~495 TFLOP/s | 50% MFU of 989.5 TFLOP/s peak bf16 |
| **M4 Max MPS** | ~5.9 TFLOP/s | 11.5M model, 764ms/step, 65K tok/step |
| **M4 Max ANE** | 600-800 GFLOP/s | 67.6M model, 99ms/step, 6-8% of 10.5 TFLOP/s peak |

---

## 4. Training Time Estimates

### 110M Model (~1.45 x 10^18 FLOPs)

| Hardware | Time | Notes |
|----------|------|-------|
| **H100** | **~49 minutes** | Single GPU, well within VRAM |
| **M4 Max MPS** | **~2.8 days** | PyTorch Metal backend |
| **M4 Max ANE** | **~24+ days** | See ANE feasibility notes below |

### 0.6B Model (~4.32 x 10^19 FLOPs)

| Hardware | Time | Notes |
|----------|------|-------|
| **H100** | **~24 hours** | Single GPU, fits in 80GB VRAM |
| **M4 Max MPS** | **~85 days** | Impractical on single Mac |
| **M4 Max ANE** | **Infeasible** | Exceeds SRAM capacity |

### Multi-GPU Scaling (H100 cluster)

| Setup | 110M | 0.6B |
|-------|------|------|
| 1x H100 | 49 min | 24 hrs |
| 4x H100 | ~13 min | ~6 hrs |
| 8x H100 | ~7 min | ~3 hrs |

> Assumes ~95% scaling efficiency for small models (communication overhead is low).

---

## 5. Memory Requirements

### Training Memory (bf16 params + fp32 optimizer states)

| Component | Per-Param Bytes | 110M | 0.6B |
|-----------|----------------|------|------|
| Model (bf16) | 2 | 220 MB | 1.2 GB |
| Gradients (bf16) | 2 | 220 MB | 1.2 GB |
| Adam momentum (fp32) | 4 | 440 MB | 2.4 GB |
| Adam variance (fp32) | 4 | 440 MB | 2.4 GB |
| **Subtotal (params)** | **12** | **1.3 GB** | **7.2 GB** |
| Activations (est.) | varies | ~2-4 GB | ~10-20 GB |
| **Total estimate** | | **~4-5 GB** | **~18-27 GB** |

### Hardware Memory Fit

| Hardware | VRAM/Memory | 110M | 0.6B |
|----------|-------------|------|------|
| H100 80GB | 80 GB HBM3 | Easily | Easily |
| M4 Max | 128 GB unified | Easily | Easily (memory) |
| ANE SRAM | 32 MB | **Hard** | **No** |
| Consumer GPU (12GB) | 12 GB GDDR6X | Yes (tight batch) | No |

---

## 6. ANE-Specific Feasibility

The ANE has unique constraints that make scaling beyond the current 67.6M config difficult:

### SRAM Wall
- **32MB on-chip SRAM** — all weights, activations, and intermediates for a single kernel must fit
- SEQ=512 is the practical maximum (SEQ=1024 barely works, 1152+ fails to compile)
- Current best: 67.6M params at NL=6, SEQ=512, DIM=768

### 110M on ANE (Karpathy 768-dim)
- 19 layers at DIM=768 — **each layer's working set exceeds what fits at SEQ=512**
- Would need SEQ=256 or lower, dramatically reducing tokens/step
- The U-curve finding (NL=6 optimal) suggests deeper models hurt on ANE
- **Verdict: Marginal.** Could work at SEQ=256 with degraded throughput. Expect ~200-400ms/step.

### 110M on ANE (maderix 1024-dim)
- 7 layers but DIM=1024 with Q_DIM=2048 — attention intermediates are large
- GQA helps reduce KV cache but Q projections are 2x wider than DIM
- **Verdict: Likely infeasible at useful sequence lengths.** The 1024 → 2048 Q projection alone strains SRAM.

### 0.6B on ANE (either architecture)
- **Verdict: Not feasible.** 27-37 layers with large intermediates far exceed SRAM capacity.
- Would require fundamental tiling/streaming work not yet implemented.

---

## 7. Architecture Trade-offs: 768 vs 1024 dim

| Factor | Karpathy (768) | maderix/ANE (1024) |
|--------|---------------|-------------------|
| **Layers for 110M** | 19 | 6-7 |
| **Layers for 0.6B** | 27 | 28-37 |
| **FFN style** | ReluSquared (non-gated, 2 matrices) | SiLU gated (3 matrices) |
| **Attention** | MHA (all heads equal) | GQA 2:1 (memory efficient) |
| **Params per layer** | 5.5M | 15.7M |
| **ANE fit** | Better (narrower tensors) | Worse (wider Q projection) |
| **GPU efficiency** | Good (standard shapes) | Better (GQA reduces KV cache) |
| **Quality/param** | Proven at small scale | Better at 0.6B+ (GQA + gated FFN) |

### Key Insight
- At **110M**, the 768-dim Karpathy arch is more practical: 19 layers gives good depth for representation, and narrower tensors fit hardware better.
- At **0.6B**, the 1024-dim maderix arch wins: GQA reduces memory for long sequences, gated FFN improves quality, and GPU utilization is better with wider tensors.

---

## 8. Training Data Requirements

| Model | Chinchilla Tokens | Dataset Size (rustbpe, ~4 bytes/token) |
|-------|-------------------|---------------------------------------|
| 110M | 2.2B | ~8.8 GB raw tokens |
| 0.6B | 12B | ~48 GB raw tokens |

**Available:** climbmix-400B provides ~631M training tokens (tokenized with rustbpe vocab=8192).
- **For 110M:** Need ~3.5x more data than available, or train for ~3.5 epochs over climbmix
- **For 0.6B:** Need ~19x more data, significant multi-epoch training or additional data sources

---

## 9. Baseline Summary for Rustane Comparison

These are the numbers to beat:

### 110M Training

| Metric | H100 | M4 MPS | M4 ANE |
|--------|------|--------|--------|
| Wall time | ~49 min | ~2.8 days | ~24+ days |
| Tokens/sec | ~750K | ~12K | ~1.5K (est.) |
| Memory | ~4-5 GB | ~4-5 GB | 32MB SRAM wall |
| Cost (cloud H100 @ $3/hr) | ~$2.50 | N/A | N/A |

### 0.6B Training

| Metric | H100 | M4 MPS | M4 ANE |
|--------|------|--------|--------|
| Wall time | ~24 hrs | ~85 days | Infeasible |
| Tokens/sec | ~280K | ~1.6K | N/A |
| Memory | ~18-27 GB | ~18-27 GB | N/A |
| Cost (cloud H100 @ $3/hr) | ~$72 | N/A | N/A |

### What Rustane Needs to Demonstrate
1. **Throughput parity or better** vs H100 PyTorch at same model size
2. **Memory efficiency** — can it train 0.6B in less memory via Rust's control?
3. **ANE utilization** — can Rust's low-level access break past the 6-8% utilization ceiling?
4. **SRAM management** — can manual tiling in Rust enable 110M+ on ANE?

---

## 10. Quick-Reference: Existing Repo Benchmarks

From actual runs in this repo (for calibration):

| Config | Params | ms/step | Steps/5min | val_bpb | Hardware |
|--------|--------|---------|------------|---------|----------|
| D8 B128 S2048 | 50.3M | ~314 | 953 | 0.998 | H100 |
| D4 B32 S2048 | 11.5M | 764 | 393 | 1.309 | M4 MPS |
| NL6 SEQ512 | 67.6M | 99 | 3,000 | 6.34 (loss) | M4 ANE |
| NL6 SEQ512 (Karpathy) | 48.8M | 164 | ~1,800 | — | M4 ANE |

> **Note:** ANE reports smoothed loss, not val_bpb. The numbers aren't directly comparable
> to MPS/H100 val_bpb due to different metrics and sequence lengths.
