# ANE SEQ=1024 Experiment — Context Length Comparison

You are an autonomous agent running ANE (Apple Neural Engine) training at SEQ=1024.

## Hypothesis

v3b achieved val_bpb=1.6347 at SEQ=512 with 72K steps (37M tokens, 4.8 hours).
SEQ=1024 will achieve better val_bpb in the same number of steps because:
- 2x context window = better language modeling (more tokens to attend to)
- 2x tokens per step = 74M tokens in 72K steps (double the data exposure)
- Direct comparison with MLX which also uses SEQ=1024

**Target:** Beat v3b's 1.6347. Stretch goal: approach MLX's 1.298.

## What's Already Done

The binary is already compiled and tested:
- Binary: `/Users/dan/Dev/autoresearch-ANE/native/build/train_karpathy_s1024`
- Step time: ~332ms/step (verified)
- All optimizer fixes included (zero-init Wo/W2, logit softcapping, separate LR groups)
- Config: NL=6, DIM=768, SEQ=1024, 48.8M params, vocab=8192

**DO NOT recompile.** The binary is ready. Just run it.

## Step 1: Launch 72K Step Run (~6.6 hours)

```bash
cd /Users/dan/Dev/autoresearch-ANE/native
./build/train_karpathy_s1024 --scratch \
  --steps 72000 \
  --lr 2.5e-4 \
  --clip 1.0 \
  --accum 2 \
  --warmup 700 \
  --data data/train_karpathy.bin \
  --val data/val_karpathy.bin \
  --token-bytes data/token_bytes.bin \
  --val-interval 5000 \
  --val-steps 20 \
  2>&1 | tee /Users/dan/Dev/autoresearch-ANE/results/overnight_s1024_v1.log
```

Settings explained:
- `--steps 72000` — same step count as v3b, cosine schedule matches
- `--lr 2.5e-4` — same stable LR as v3b (half of sweep winner, proven stable for 72K)
- `--accum 2` — same as v3b winner
- `--warmup 700` — 1% of steps (same ratio as v3b's 1000/72000 ≈ 700/72000)
- `--val-interval 5000` — validation every ~28 minutes

## Step 2: Monitor

Check periodically:

```bash
# Latest val_bpb
grep "\[VAL" /Users/dan/Dev/autoresearch-ANE/results/overnight_s1024_v1.log | tail -5

# Activation health
grep "^step" /Users/dan/Dev/autoresearch-ANE/results/overnight_s1024_v1.log | tail -3
```

**Compare against v3b at same step counts:**

| Step | v3b (SEQ=512) | s1024 (this run) |
|------|---------------|------------------|
| 5K   | 2.1539        |                  |
| 10K  | 2.0525        |                  |
| 15K  | 1.9721        |                  |
| 20K  | 1.9162        |                  |
| 25K  | 1.8669        |                  |
| 30K  | 1.8257        |                  |
| 35K  | ~1.79         |                  |
| 40K  | ~1.76         |                  |
| 50K  | ~1.70         |                  |
| 60K  | ~1.66         |                  |
| 72K  | 1.6347        |                  |

Fill in this table as validation checkpoints come in. Report whether SEQ=1024 is winning or losing vs SEQ=512 at each checkpoint.

**Stability:** If activations exceed x[±30], the run may be diverging. Note the step number. If x exceeds ±50, kill it and report.

## Step 3: After Completion

When the run finishes:

1. Extract final val_bpb:
```bash
grep "\[VAL" /Users/dan/Dev/autoresearch-ANE/results/overnight_s1024_v1.log | tail -1
```

2. Log result:
```bash
echo -e "s1024_v1\t<VAL_BPB>\tNL6_SEQ1024_LR2.5e-4_ACC2_WU700_72K\tcomplete\tSEQ=1024 vs v3b SEQ=512 comparison" >> /Users/dan/Dev/autoresearch-ANE/results/ane_karpathy_results.tsv
```

3. Print summary comparison:
```
=== SEQ=1024 vs SEQ=512 Comparison ===
SEQ=512  (v3b):  val_bpb = 1.6347 (72K steps, 37M tokens, 4.8 hours)
SEQ=1024 (this): val_bpb = X.XXXX (72K steps, 74M tokens, ~6.6 hours)
MLX target:      val_bpb = 1.298  (SEQ=1024, 15.7M params)
```

4. **Decision point:** Based on the result:
   - If SEQ=1024 beats v3b significantly (>0.1 bpb better): recommend a longer SEQ=1024 run (200K+ steps)
   - If SEQ=1024 is similar to v3b: context length isn't the bottleneck, longer SEQ=512 run is better (faster steps)
   - If SEQ=1024 is worse: SRAM spill overhead kills the benefit, stick with SEQ=512

## What NOT to Do

- Do NOT recompile the binary
- Do NOT modify train.m or any source files
- Do NOT change the model config
- Do NOT run multiple training processes (only one can use ANE at a time)
- Do NOT change --steps mid-run

## Key Context

- ANE is exclusive hardware: only one process can use it at a time
- v3b's binary (SEQ=512) is at build/train_dynamic
- This binary (SEQ=1024) is at build/train_karpathy_s1024
- Both binaries include the same optimizer fixes (zero-init, softcapping, separate LR)
- Results go in: ~/Dev/autoresearch-ANE/results/

## START

Launch the 72K step run immediately. Monitor and fill in the comparison table as checkpoints arrive.
