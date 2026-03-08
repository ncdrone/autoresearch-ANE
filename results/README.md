# Experiment Results — autoresearch-ANE

## Hardware
- Apple M4 Max, 128GB unified memory
- 40 GPU cores (MPS), 16 ANE cores, ~32MB ANE SRAM

## Directory Structure

### `mps_pretest/`
MPS/Metal GPU pre-test sweep results from autoresearch-macos.
- Depth sweep: D={2,4,6,8} at batch 16, total 65K
- Batch sweep: B={16,32,64,128,256} at depth 4
- Winner: **Depth 4, Batch 32, val_bpb = 1.309**
- Key finding: More steps > bigger batches on Metal (opposite of GPU)

### `ane_bench_*.tsv` + `logs_*/`
ANE hardware profiling — 15 configs (NL={2,4,8,16,24} × SEQ={256,512,1024}).
- All 15 compile and train successfully
- SRAM wall at SEQ=1024 (SEQ=1152+ fails)
- 6-8% ANE utilization (600-800 GFLOP/s of 10.5 TFLOP/s peak)
- Timing breakdown: 33% ANE compute, 30% IO, 37% CPU

### `sweep_5min/`
ANE 5-minute training sweep with real climbmix data.
- 9 configs over 2 rounds, each trained for ~5 minutes
- First real loss numbers on ANE with real data
- Winner: **NL=6, SEQ=512, 67.6M params, 99ms/step, smoothed loss 6.340**
- Clear U-shaped depth curve at SEQ=512: NL=4 (6.74) → NL=6 (6.34) → NL=8 (6.94) → NL=10 (6.79) → NL=12 (7.14)
- Key finding: more steps > bigger model (same as MPS)

## Key Numbers

| System | Params | ms/step | Steps/5min | Config |
|--------|--------|---------|------------|--------|
| H100 (Karpathy) | 50.3M | ~314ms | 953 | D8, B128, S2048 |
| M4 Max MPS | 11.5M | 764ms | 393 | D4, B32, S2048 |
| M4 Max ANE | 67.6M | 99ms | 3,000 | NL=6, SEQ=512 |

ANE trains a 6x bigger model 8x faster than MPS on the same chip.
