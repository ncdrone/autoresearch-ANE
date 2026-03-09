#!/bin/bash
# ANE Karpathy Pretests — run all 7 pretests sequentially
# Usage: cd native && bash scripts/run_pretests.sh
set -e

OUTDIR="/Users/dan/Dev/autoresearch-ANE/results/pretests_karpathy"
mkdir -p "$OUTDIR"

COMMON="--scratch --lr 2e-4 --clip 1.0 --data data/train_karpathy.bin --val data/val_karpathy.bin --token-bytes data/token_bytes.bin --val-interval 500 --val-steps 20"

echo "=== P2: accum=1 ==="
timeout 300 ./build/train_dynamic $COMMON --steps 1800 --accum 1 2>&1 | tee "$OUTDIR/P2_accum1.log"

echo "=== P3: accum=5 ==="
timeout 300 ./build/train_dynamic $COMMON --steps 1800 --accum 5 2>&1 | tee "$OUTDIR/P3_accum5.log"

echo "=== P4: accum=20 ==="
timeout 300 ./build/train_dynamic $COMMON --steps 1800 --accum 20 2>&1 | tee "$OUTDIR/P4_accum20.log"

echo "=== P5: NL=4 (recompile) ==="
xcrun clang -O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK -DNLAYERS=4 \
  -I. -Iruntime -Imil -Itraining -include training/models/gpt_karpathy.h \
  -o build/train_karpathy_nl4 training/train.m -ldl \
  -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate
timeout 300 ./build/train_karpathy_nl4 $COMMON --steps 1800 --accum 10 2>&1 | tee "$OUTDIR/P5_nl4.log"

echo "=== P6: NL=8 (recompile) ==="
xcrun clang -O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK -DNLAYERS=8 \
  -I. -Iruntime -Imil -Itraining -include training/models/gpt_karpathy.h \
  -o build/train_karpathy_nl8 training/train.m -ldl \
  -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate
timeout 300 ./build/train_karpathy_nl8 $COMMON --steps 1800 --accum 10 2>&1 | tee "$OUTDIR/P6_nl8.log"

echo "=== P7: SEQ=1024, NL=6 (recompile) ==="
xcrun clang -O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK -DSEQ=1024 \
  -I. -Iruntime -Imil -Itraining -include training/models/gpt_karpathy.h \
  -o build/train_karpathy_s1024 training/train.m -ldl \
  -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate
timeout 300 ./build/train_karpathy_s1024 $COMMON --steps 900 --accum 10 2>&1 | tee "$OUTDIR/P7_seq1024.log"

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Baseline (10K run, step 1500): val_bpb=2.6002"
echo ""
for f in "$OUTDIR"/P*.log; do
  name=$(basename "$f" .log)
  bpb=$(grep "\[VAL" "$f" | tail -1 | grep -oP 'val_bpb=\K[0-9.]+')
  steps=$(grep "^step" "$f" | tail -1 | awk '{print $2}')
  echo "$name: val_bpb=$bpb (last step ~$steps)"
done
