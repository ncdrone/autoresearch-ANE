# Makefile — Build ANE native code for Apple Silicon
# Requires: macOS 15+, Xcode command line tools, Apple Silicon

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-unused-function -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate
LDFLAGS = -ldl $(FRAMEWORKS)

# Model selection: make MODEL=stories110m (or qwen3_06b, gpt_autoresearch)
MODEL ?= gpt_autoresearch
MODEL_HEADER = training/models/$(MODEL).h

# Shared include paths
INCLUDES = -I. -Iruntime -Imil -Itraining -include $(MODEL_HEADER)

# === Bridge (shared library for Python ctypes) ===
.PHONY: bridge
bridge: build/libane_bridge.dylib

build/libane_bridge.dylib: bridge/ane_bridge.m bridge/ane_bridge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -shared -o $@ bridge/ane_bridge.m $(LDFLAGS) -install_name @rpath/libane_bridge.dylib

# === Probes (standalone executables) ===
.PHONY: probes
probes: build/sram_bench build/sram_probe build/api_exploration build/inmem_basic build/inmem_bench build/inmem_peak

build/sram_bench: probes/sram_bench.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build/sram_probe: probes/sram_probe.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build/api_exploration: probes/api_exploration.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build/inmem_basic: probes/inmem_basic.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build/inmem_bench: probes/inmem_bench.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build/inmem_peak: probes/inmem_peak.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# === Weight experiments ===
.PHONY: weight-tests
weight-tests: build/test_weight_reload build/test_dynamic_matmul

build/test_weight_reload: probes/test_weight_reload.m
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# test_weight_patch and test_dynamic_matmul need the training includes
build/test_weight_patch: probes/test_weight_patch.m
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

build/test_dynamic_matmul: probes/test_dynamic_matmul.m
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# === Dynamic training ===
.PHONY: train
train: build/train_dynamic

build/train_dynamic: training/train.m mil/mil_dynamic.h training/cpu_ops.h runtime/config.h runtime/io.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ training/train.m $(LDFLAGS)

# === All targets ===
.PHONY: all
all: bridge probes weight-tests train

# === Clean ===
.PHONY: clean
clean:
	rm -rf build

# === Quick test: verify ANE is accessible ===
.PHONY: test-ane
test-ane: build/inmem_basic
	@echo "=== Testing ANE access ==="
	./build/inmem_basic

# === Run SRAM benchmark ===
.PHONY: bench-sram
bench-sram: build/sram_bench build/sram_probe
	@echo "=== SRAM Benchmark ==="
	./build/sram_bench
	@echo ""
	@echo "=== Fine-grained SRAM Probe ==="
	./build/sram_probe

# === Run weight experiment suite ===
.PHONY: bench-weights
bench-weights: build/test_weight_reload build/test_dynamic_matmul
	@echo "=== Weight Reload Test ==="
	./build/test_weight_reload
	@echo ""
	@echo "=== Dynamic Matmul Benchmark ==="
	./build/test_dynamic_matmul

.PHONY: help
help:
	@echo "ANE Native Build System"
	@echo ""
	@echo "Targets:"
	@echo "  bridge          Build libane_bridge.dylib (Python ctypes interface)"
	@echo "  probes          Build all probe/benchmark executables"
	@echo "  weight-tests    Build weight patching experiments"
	@echo "  train           Build dynamic ANE training loop"
	@echo "  all             Build everything"
	@echo ""
	@echo "Run targets:"
	@echo "  test-ane        Quick test: verify ANE hardware access"
	@echo "  bench-sram      Probe ANE SRAM size and performance cliffs"
	@echo "  bench-weights   Test weight update strategies"
	@echo ""
	@echo "Options:"
	@echo "  MODEL=xxx       Select model config (default: gpt_autoresearch)"
	@echo "                  Options: stories110m, qwen3_06b, gpt_autoresearch"
