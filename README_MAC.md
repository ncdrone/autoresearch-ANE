# autoresearch-ANE: Apple Silicon Deep Research Edition

Autonomous LLM pretraining research on Apple Silicon, with direct Apple Neural Engine (ANE) access via ported [maderix/ANE](https://github.com/maderix/ANE) native code.

**Goal**: Unlock the full compute potential of Apple Silicon by using ALL hardware units (ANE + GPU + CPU + AMX) for ML training, not just the GPU.

## What's Here

This project has two layers:

1. **Python training** (`train_mac.py`) — Standard PyTorch MPS training, works today
2. **Native ANE research** (`native/`) — Ported Objective-C code for direct ANE hardware access, probing, and experimental training

## Quick Start

### Training (works now)
```bash
cp pyproject_mac.toml pyproject.toml && uv sync
uv run prepare.py --num-shards 8
uv run train_mac.py
```

### ANE Research (requires macOS on Apple Silicon)
```bash
# Build everything
cd native && make all

# Quick test: verify ANE hardware access
make test-ane

# Probe ANE SRAM size (find the performance cliff)
make bench-sram

# Test dynamic weights (can we avoid recompilation?)
make bench-weights

# Or use Python harness
cd .. && python ane_benchmark.py --all
```

## Architecture

### Training: MPS Backend (`train_mac.py`)

| Feature | CUDA (Original) | Mac Silicon (This Fork) |
|---------|-----------------|------------------------|
| GPU | NVIDIA H100 | Apple Silicon GPU (MPS) |
| Attention | Flash Attention 3 | PyTorch native SDPA |
| Precision | bfloat16 | float16 |
| Memory | 80GB VRAM | Up to 192GB unified |

### ANE Research: Native Code (`native/`)

The `native/` directory contains ported and adapted code from [maderix/ANE](https://github.com/maderix/ANE) — the project that proved ANE hardware can do ML training, not just inference.

#### Key Breakthrough: Dynamic Weights

The original maderix/ANE project achieved only 5-9% ANE utilization because weights were baked into compiled kernels. Every optimizer step required recompiling ~60 kernels (~3.7s overhead).

The `training_dynamic/` pipeline solves this: **weights are packed into the input IOSurface alongside activations**. Kernels compile ONCE at startup. Weight updates are just memcpy to shared memory.

```
Static pipeline:  compile(W) → eval(x) → update(W) → RECOMPILE(W) → eval(x) → ...
Dynamic pipeline: compile()  → write(x,W) → eval() → write(x,W') → eval() → ...
```

#### Native Directory Structure

```
native/
├── bridge/              # Python-callable C interface via ctypes
│   ├── ane_bridge.h     # C API header
│   └── ane_bridge.m     # Obj-C implementation
├── runtime/             # Core ANE interface
│   ├── ane_runtime.h    # dlopen + compile + IOSurface + eval
│   ├── config.h         # Model structs, derived sizes, ANE init
│   └── io.h             # IOSurface I/O, NEON f16/f32 conversion
├── mil/                 # MIL (Model Intermediate Language) code generation
│   ├── mil_dynamic.h    # Dynamic weight kernels (the breakthrough)
│   └── ane_mil_gen.h    # Static weight kernels (reference)
├── training/            # ANE training loop
│   ├── train.m          # Dynamic weight training (compile-once)
│   ├── cpu_ops.h        # CPU: RMSNorm, Adam, cross-entropy (Accelerate)
│   └── models/          # Model configs
│       ├── gpt_autoresearch.h  # Our GPT config
│       ├── stories110m.h       # Stories110M (Llama2-style)
│       └── qwen3_06b.h         # Qwen3-0.6B (GQA)
├── probes/              # Hardware exploration and benchmarks
│   ├── sram_bench.m     # Probe ANE SRAM size
│   ├── sram_probe.m     # Fine-grained SRAM probing
│   ├── api_exploration.m        # Discover private ANE APIs
│   ├── test_weight_patch.m      # 6 approaches to avoid recompilation
│   ├── test_weight_reload.m     # Unload/reload weight test
│   ├── test_dynamic_matmul.m    # Dynamic matmul benchmark
│   ├── inmem_basic.m    # Basic in-memory compilation test
│   ├── inmem_bench.m    # In-memory compilation benchmark
│   └── inmem_peak.m     # Peak throughput measurement
└── Makefile             # Build system
```

## Python Interface

### `ane_bridge.py` — Real ctypes bridge

```python
from ane_bridge import ANEBridge

bridge = ANEBridge()
print(bridge.get_info())  # Hardware detection always works

if bridge.native_available:
    # Generate dynamic matmul MIL (weights via IOSurface, no recompile)
    mil = ANEBridge.gen_dynamic_matmul_mil(ic=768, oc=768, seq=256)

    # Compile once, use forever
    kernel = bridge.compile_kernel(mil, input_sizes=[...], output_sizes=[...])

    # Pack activations + weights into input, eval on ANE
    bridge.write_input(kernel, 0, packed_data)
    bridge.eval(kernel)
    output = bridge.read_output(kernel, 0, output_bytes)
```

### `ane_benchmark.py` — Probe everything

```bash
python ane_benchmark.py --basic     # Verify ANE access
python ane_benchmark.py --sram      # Find SRAM performance cliff
python ane_benchmark.py --dynamic   # Test dynamic weight strategies
python ane_benchmark.py --compare   # ANE vs MPS vs CPU
python ane_benchmark.py --explore   # Discover private APIs
python ane_benchmark.py --all       # Run everything
```

## Research Questions

This project continues the research started by maderix/ANE:

1. **SRAM boundaries**: Where exactly does ANE performance cliff? How does this vary by chip (M1 vs M4)?
2. **Dynamic weight overhead**: How much throughput do we lose by packing weights into IOSurface vs constants?
3. **Weight reload API**: Does `_ANEWeight` + `weightsBuffer` in `_ANERequest` work for runtime weight swapping?
4. **Memory scan patching**: Can we find and patch compiled weights in process memory?
5. **Multi-unit pipeline**: Can we pipeline ANE + GPU + CPU + AMX to exceed any single unit's throughput?
6. **Compilation limit**: What causes the ~119 compile limit? Is there a cleanup API?
7. **Undiscovered ops**: What ANE operations exist beyond what's been tested?

## Apple Silicon Compute Units

| Unit | Peak (M4 Max) | Used For (Currently) | Potential |
|------|---------------|---------------------|-----------|
| **ANE** | 38 TOPS | CoreML inference only | Training (proven by maderix/ANE) |
| **GPU** | 15.2 TFLOPS | MPS training | Could pipeline with ANE |
| **CPU** | P+E cores | Everything else | Accelerate/vDSP for element-wise |
| **AMX** | ~2 TFLOPS | BLAS calls | Matrix operations without GPU |

**The opportunity**: Today's ML frameworks use GPU only. With 128GB unified memory and 4 compute units sharing the same address space, Apple Silicon should be able to exceed GPU-only throughput.

## Files

| File | Purpose |
|------|---------|
| `train_mac.py` | PyTorch MPS training (production) |
| `ane_bridge.py` | Python ctypes bridge to native ANE code |
| `ane_benchmark.py` | Benchmark suite for all compute units |
| `ane_inference.py` | CoreML inference on ANE |
| `convert_to_coreml.py` | PyTorch → CoreML model conversion |
| `native/` | Ported maderix/ANE Objective-C code |
| `prepare.py` | Data preparation (read-only) |
| `train.py` | Original CUDA training (upstream reference) |

## Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — The autonomous research framework and climbmix-400b dataset
- [maderix/ANE](https://github.com/maderix/ANE) — Pioneering reverse-engineering of ANE for training. The `native/` directory is ported from this project.
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — macOS fork with Metal/MPS support that bridged Karpathy's CUDA code to Apple Silicon
