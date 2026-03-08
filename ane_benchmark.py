"""
ANE Benchmark Suite — Probe Apple Silicon compute capabilities

Run on your Mac to discover:
  1. ANE SRAM size (where performance cliffs happen)
  2. Whether dynamic weights work (no recompilation needed)
  3. Peak ANE throughput for matmul at various sizes
  4. Comparison: ANE vs MPS vs CPU for your specific chip

Usage:
    python ane_benchmark.py --all          # Run everything
    python ane_benchmark.py --sram         # SRAM size probe
    python ane_benchmark.py --dynamic      # Dynamic weight tests
    python ane_benchmark.py --compare      # ANE vs MPS vs CPU

Requires: macOS 15+, Apple Silicon, native code built (cd native && make all)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np


def run_native_probe(name, timeout=120):
    """Build and run a native probe, return stdout."""
    native_dir = Path(__file__).parent / "native"
    exe = native_dir / "build" / name

    # Build if needed
    if not exe.exists():
        print(f"  Building {name}...")
        result = subprocess.run(
            ["make", f"build/{name}"],
            cwd=str(native_dir),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Build failed: {result.stderr[:500]}")
            return None

    # Run
    result = subprocess.run(
        [str(exe)],
        capture_output=True, text=True,
        timeout=timeout
    )
    if result.returncode != 0:
        print(f"  Run failed (exit {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
    return result.stdout


def bench_sram():
    """Probe ANE SRAM size by running convolutions with increasing weight sizes."""
    print("=" * 60)
    print("ANE SRAM Probe")
    print("=" * 60)
    print()
    print("Running 1x1 convolutions with increasing weight matrices.")
    print("When weights exceed SRAM, performance drops sharply.")
    print()

    output = run_native_probe("sram_bench")
    if output:
        print(output)

    print()
    print("Fine-grained probe:")
    print()
    output = run_native_probe("sram_probe")
    if output:
        print(output)


def bench_dynamic_weights():
    """Test dynamic weight strategies — can we avoid recompilation?"""
    print("=" * 60)
    print("Dynamic Weight Tests")
    print("=" * 60)
    print()
    print("Testing whether weights can be updated without recompiling")
    print("ANE kernels. This is the key to practical ANE training.")
    print()

    print("--- Test 1: Weight Reload (unload + overwrite + reload) ---")
    output = run_native_probe("test_weight_reload", timeout=60)
    if output:
        print(output)

    print()
    print("--- Test 2: Dynamic Matmul (weights via IOSurface input) ---")
    output = run_native_probe("test_dynamic_matmul", timeout=120)
    if output:
        print(output)


def bench_ane_basic():
    """Basic ANE connectivity test."""
    print("=" * 60)
    print("ANE Basic Test")
    print("=" * 60)
    print()
    output = run_native_probe("inmem_basic")
    if output:
        print(output)
    else:
        print("FAILED — ANE not accessible. Check macOS version and SIP.")


def bench_ane_peak():
    """Peak ANE throughput benchmark."""
    print("=" * 60)
    print("ANE Peak Throughput")
    print("=" * 60)
    print()
    output = run_native_probe("inmem_peak", timeout=180)
    if output:
        print(output)


def bench_mps():
    """Benchmark MPS (Metal GPU) for comparison."""
    print("=" * 60)
    print("MPS (Metal GPU) Benchmark")
    print("=" * 60)
    print()

    try:
        import torch
        if not torch.backends.mps.is_available():
            print("MPS not available")
            return
    except ImportError:
        print("PyTorch not installed — skipping MPS benchmark")
        return

    device = torch.device("mps")

    sizes = [(256, 256, 64), (512, 512, 64), (768, 768, 256),
             (1024, 1024, 256), (2048, 2048, 256)]

    print(f"{'M':>6} x {'K':>6} x {'N':>6}  {'ms':>8}  {'TFLOPS':>8}")
    print("-" * 50)

    for M, K, N in sizes:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            torch.mm(A, B)
        torch.mps.synchronize()

        # Benchmark
        iters = 100
        t0 = time.time()
        for _ in range(iters):
            torch.mm(A, B)
        torch.mps.synchronize()
        elapsed = (time.time() - t0) / iters * 1000

        flops = 2 * M * K * N
        tflops = flops / (elapsed * 1e9)
        print(f"{M:6d} x {K:6d} x {N:6d}  {elapsed:8.3f}  {tflops:8.2f}")


def bench_cpu():
    """Benchmark CPU (Accelerate/BLAS) for comparison."""
    print("=" * 60)
    print("CPU (Accelerate) Benchmark")
    print("=" * 60)
    print()

    sizes = [(256, 256, 64), (512, 512, 64), (768, 768, 256),
             (1024, 1024, 256), (2048, 2048, 256)]

    print(f"{'M':>6} x {'K':>6} x {'N':>6}  {'ms':>8}  {'GFLOPS':>8}")
    print("-" * 50)

    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Warmup
        for _ in range(3):
            np.dot(A, B)

        # Benchmark
        iters = 20
        t0 = time.time()
        for _ in range(iters):
            np.dot(A, B)
        elapsed = (time.time() - t0) / iters * 1000

        flops = 2 * M * K * N
        gflops = flops / (elapsed * 1e6)
        print(f"{M:6d} x {K:6d} x {N:6d}  {elapsed:8.3f}  {gflops:8.1f}")


def compare_all():
    """Run ANE vs MPS vs CPU comparison."""
    print("=" * 60)
    print("Compute Backend Comparison")
    print("=" * 60)
    print()

    from ane_bridge import ANEBridge
    bridge = ANEBridge()
    info = bridge.get_info()
    print(f"Chip: {info['chip']}  |  Memory: {info['memory_gb']:.0f}GB")
    print(f"ANE: {info['ane_tops']} TOPS  |  GPU: {info['gpu_tflops']} TFLOPS")
    print()

    bench_ane_basic()
    print()
    bench_mps()
    print()
    bench_cpu()


def explore_api():
    """Run the API exploration tool to discover ANE capabilities."""
    print("=" * 60)
    print("ANE API Exploration")
    print("=" * 60)
    print()
    print("Probing private AppleNeuralEngine.framework for:")
    print("  - Available classes and methods")
    print("  - MLModelAsset in-memory loading")
    print("  - MIL compilation pipeline")
    print("  - CoreML -> ANE dispatch path")
    print()

    output = run_native_probe("api_exploration", timeout=60)
    if output:
        print(output)


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("This benchmark suite requires macOS on Apple Silicon.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="ANE Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--basic", action="store_true", help="Basic ANE connectivity test")
    parser.add_argument("--sram", action="store_true", help="Probe ANE SRAM size")
    parser.add_argument("--dynamic", action="store_true", help="Test dynamic weight strategies")
    parser.add_argument("--peak", action="store_true", help="Peak ANE throughput")
    parser.add_argument("--mps", action="store_true", help="MPS (Metal GPU) benchmark")
    parser.add_argument("--cpu", action="store_true", help="CPU benchmark")
    parser.add_argument("--compare", action="store_true", help="ANE vs MPS vs CPU comparison")
    parser.add_argument("--explore", action="store_true", help="API exploration")
    args = parser.parse_args()

    if args.all:
        bench_ane_basic()
        print()
        bench_sram()
        print()
        bench_dynamic_weights()
        print()
        bench_ane_peak()
        print()
        bench_mps()
        print()
        bench_cpu()
    elif args.basic:
        bench_ane_basic()
    elif args.sram:
        bench_sram()
    elif args.dynamic:
        bench_dynamic_weights()
    elif args.peak:
        bench_ane_peak()
    elif args.mps:
        bench_mps()
    elif args.cpu:
        bench_cpu()
    elif args.compare:
        compare_all()
    elif args.explore:
        explore_api()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. cd native && make all     # Build native ANE code")
        print("  2. python ane_benchmark.py --basic  # Verify ANE access")
        print("  3. python ane_benchmark.py --all    # Run everything")
