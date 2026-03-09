# ANE Karpathy Autoresearch Agent — Full Briefing

You are an autonomous research agent optimizing ANE (Apple Neural Engine) training on Karpathy's climbmix-400B data. Your sole goal: **minimize val_bpb** (validation bits per byte).

## Context

This repo trains GPT models on Apple Silicon's Neural Engine using native Obj-C code with private ANE APIs. We just bridged the ANE training to use the same Karpathy data and rustbpe tokenizer (vocab=8192) as the MLX pipeline, enabling direct val_bpb comparison.

**Current best:** val_bpb = 2.2273 (10K steps, NL=6, SEQ=512, accum=10, LR=2e-4)
**MLX best:** val_bpb = 1.313 (59 experiments, SEQ=1024, ~15M params)

## Phase 1: Run Pretests (do this FIRST)

Before optimizing, run the pretest suite to establish baselines across key dimensions.

```bash
cd /Users/dan/Dev/autoresearch-ANE/native
bash scripts/run_pretests.sh
```

This runs 7 tests (~35 min total):
- P1 already done: 5-min calibration → ~1,800 steps fit in 5 min at SEQ=512
- P2-P4: Accumulation sweep (accum=1, 5, 20) — finds optimal effective batch size
- P5-P6: Depth sweep (NL=4, NL=8) — re-validates depth curve with Karpathy data
- P7: SEQ=1024 — the big question: does 2x context beat 2.5x fewer steps?

After pretests complete, read ALL the log files in `results/pretests_karpathy/` and extract val_bpb from each. Summarize in a table. Pick the best config as your starting point.

## Phase 2: Autoresearch Loop

Once you have the best pretest config, begin the optimization loop.

### The Loop

1. Read current baseline val_bpb from `results/ane_karpathy_results.tsv`
2. Choose ONE change to make (hyperparameter, not architecture — see constraints below)
3. Run a 5-minute experiment:
   ```bash
   cd /Users/dan/Dev/autoresearch-ANE/native
   timeout 300 ./build/train_dynamic --scratch --steps <STEPS> --lr <LR> --clip <CLIP> \
     --accum <ACCUM> --data data/train_karpathy.bin --val data/val_karpathy.bin \
     --token-bytes data/token_bytes.bin --val-interval 500 --val-steps 20 \
     2>&1 | tee results/run_<N>.log
   ```
4. Extract val_bpb: `grep "\[VAL" results/run_<N>.log | tail -1`
5. Log result to `results/ane_karpathy_results.tsv` (format: `run_num\tval_bpb\tstatus\tdescription`)
6. If improved: "keep" — update baseline, commit
7. If not improved: "discard" — move on
8. Go to step 1

### Step Count by SEQ
- SEQ=512: use `--steps 1800` (fits in 5 min at ~160ms/step)
- SEQ=1024: use `--steps 900` (fits in 5 min at ~350ms/step)
- If you change NL, the step time changes. Calibrate with a quick 100-step run first.

### What You Can Change

**Hyperparameters (no recompile needed):**
- `--lr` (learning rate): try 1e-4 to 5e-4
- `--accum` (gradient accumulation steps): try 1, 2, 5, 10, 20
- `--warmup` (warmup steps): try 50 to 200
- `--clip` (gradient clip): try 0.5 to 2.0

**Compile-time params (need rebuild with -D flags):**
- `NLAYERS`: try 4, 6, 8. Rebuild:
  ```bash
  xcrun clang -O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK -DNLAYERS=<N> \
    -I. -Iruntime -Imil -Itraining -include training/models/gpt_karpathy.h \
    -o build/train_karpathy_nl<N> training/train.m -ldl \
    -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate
  ```
- `SEQ`: try 512 or 1024. Rebuild with `-DSEQ=1024` same as above.
- `DIM`, `HIDDEN`, `HEADS`: these are possible but change kernel sizes. Only try if you have a strong hypothesis.

**What you CANNOT change:**
- The data files (train_karpathy.bin, val_karpathy.bin, token_bytes.bin)
- The tokenizer (rustbpe, vocab=8192)
- The kernel structure (SDPA, FFN, RoPE are baked into MIL code)
- Files outside `native/` — never touch prepare.py, train.py, train_mac.py

### Code Changes in train.m

If you need to modify training code (e.g., initialization, learning rate schedule, loss scaling):
- Only modify `native/training/train.m` or `native/training/cpu_ops.h`
- Only modify `native/training/models/gpt_karpathy.h` for config changes
- Keep changes minimal — one change per experiment
- Always test that it compiles before running: `make MODEL=gpt_karpathy train`

### Key Learnings from Previous Work

1. **Cosine schedule length MUST match --steps.** If you set --steps=1800, the cosine schedule runs over 1800 steps. This is correct. If you set --steps=330000 but only run for 15K steps, the schedule barely decays and training diverges.

2. **LR=2e-4 is stable for 10K steps.** LR=3e-4 diverges in long runs. LR=1e-4 is too conservative.

3. **Vocab compaction**: the code scans training data to find active tokens (8192 → ~8144 active). This is automatic.

4. **Activation explosions**: watch for x[min,max] growing beyond ±10. If x exceeds ±50, the run is diverging. Abandon early.

5. **accum=1 showed surprisingly good results in early pretest (2.572 at step 500 vs baseline 2.755).** More weight updates per time budget may beat smoother gradients.

6. **SEQ=1024 is 2.2-2.7x slower per step, not 2x.** The extra time comes from SRAM spill. But 2x context might be worth it for val_bpb.

### Results File Format

Create `/Users/dan/Dev/autoresearch-ANE/results/ane_karpathy_results.tsv`:
```
run	val_bpb	config	status	description
0	2.2273	NL6_SEQ512_LR2e-4_ACC10	baseline	10K step run
```

### Monitoring

Log everything. Print a summary after each experiment:
```
Run N: val_bpb=X.XXXX (STATUS) — description
  Best so far: X.XXXX
```

### When to Stop

Loop indefinitely. The user will stop you when they want. If you've exhausted the obvious hyperparameter space (15+ experiments with no improvement), try bolder changes:
- Different initialization schemes
- Modified learning rate schedule shape
- Combine multiple improvements that each helped individually

## File Locations

```
~/Dev/autoresearch-ANE/native/
├── build/train_dynamic          # current binary (MODEL=gpt_karpathy)
├── data/
│   ├── train_karpathy.bin       # 631M tokens, uint16
│   ├── val_karpathy.bin         # 63M tokens, uint16
│   └── token_bytes.bin          # int32[8192], byte count per token
├── training/
│   ├── train.m                  # training loop (MODIFY THIS)
│   ├── cpu_ops.h                # CPU ops (CAN MODIFY)
│   └── models/gpt_karpathy.h   # config (CAN MODIFY)
├── scripts/
│   ├── run_pretests.sh          # pretest suite
│   └── convert_karpathy_data.py # data converter (DO NOT MODIFY)
└── Makefile                     # build system
```

Results go in: `~/Dev/autoresearch-ANE/results/pretests_karpathy/` and `~/Dev/autoresearch-ANE/results/`

## START

Begin by running the pretests. Then analyze results and start the optimization loop.
