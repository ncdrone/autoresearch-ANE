"""
Direct Apple Neural Engine (ANE) Bridge via ctypes

This module provides Python access to ANE hardware through native Objective-C
code compiled from the maderix/ANE research project. Unlike CoreML, this
bypasses Apple's public APIs and talks directly to the ANE via private
AppleNeuralEngine.framework.

Two modes:
  1. Native bridge (requires building native/): Full ANE access via ctypes
     to libane_bridge.dylib — compile MIL programs, run kernels, read/write
     IOSurface tensors.
  2. Fallback mode: Hardware detection and MIL generation only (no native build).

Architecture (from maderix/ANE dynamic training pipeline):
  - Kernels compiled ONCE at startup from MIL text
  - Weights passed as part of input IOSurface (no recompilation on update!)
  - Data layout: [1, C, 1, S] fp16 channel-first (ANE native format)
  - Forward + backward on ANE, weight gradients + optimizer on CPU

Build native code first:
    cd native && make bridge

Usage:
    from ane_bridge import ANEBridge

    bridge = ANEBridge()
    if bridge.native_available:
        # Compile and run a kernel directly on ANE
        kernel = bridge.compile_kernel(mil_text, weight_data, ...)
        bridge.write_input(kernel, 0, input_data)
        bridge.eval(kernel)
        output = bridge.read_output(kernel, 0, output_size)
    else:
        # Still useful for hardware detection and MIL generation
        print(bridge.get_info())

WARNING: Uses private macOS APIs. May break across macOS versions.
"""

import os
import sys
import ctypes
import struct
import subprocess
import platform
import numpy as np
from pathlib import Path


# ANE TOPS estimates by chip generation
ANE_TOPS = {
    "M1": 11.0, "M1 Pro": 11.0, "M1 Max": 11.0, "M1 Ultra": 22.0,
    "M2": 15.8, "M2 Pro": 15.8, "M2 Max": 15.8, "M2 Ultra": 31.6,
    "M3": 18.0, "M3 Pro": 18.0, "M3 Max": 18.0, "M3 Ultra": 36.0,
    "M4": 38.0, "M4 Pro": 38.0, "M4 Max": 38.0, "M4 Ultra": 76.0,
}

# GPU TFLOPS estimates (Metal fp16)
GPU_TFLOPS = {
    "M1": 2.6, "M1 Pro": 5.2, "M1 Max": 10.4, "M1 Ultra": 20.8,
    "M2": 3.6, "M2 Pro": 6.8, "M2 Max": 13.6, "M2 Ultra": 27.2,
    "M3": 3.6, "M3 Pro": 7.0, "M3 Max": 14.2, "M3 Ultra": 28.4,
    "M4": 3.8, "M4 Pro": 7.6, "M4 Max": 15.2, "M4 Ultra": 30.4,
}


def check_ane_available():
    """Check if ANE hardware is available on this system."""
    if sys.platform != "darwin":
        return False
    if platform.machine() != "arm64":
        return False
    try:
        result = subprocess.run(
            ["ioreg", "-l", "-w", "0"],
            capture_output=True, text=True, timeout=10
        )
        return "ANECompiler" in result.stdout or "appleane" in result.stdout.lower()
    except Exception:
        return False


def detect_chip():
    """Detect Apple Silicon chip model."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        brand = result.stdout.strip()
    except Exception:
        brand = "unknown"

    chip = "unknown"
    for name in sorted(ANE_TOPS.keys(), key=len, reverse=True):
        if name.replace(" ", "") in brand.replace(" ", ""):
            chip = name
            break
    return chip, brand


def get_memory_gb():
    """Get total system memory in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip()) / (1024**3)
    except Exception:
        return 0


def build_weight_blob(weights_f32, rows, cols):
    """
    Build ANE weight blob from float32 numpy array.

    ANE weight blob format:
      - 128-byte header (magic numbers, sizes)
      - fp16 weight data
    """
    if isinstance(weights_f32, np.ndarray):
        weights_f32 = weights_f32.flatten()
    assert len(weights_f32) == rows * cols

    wsize = rows * cols * 2  # fp16
    total = 128 + wsize

    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64] = 0xEF; header[65] = 0xBE; header[66] = 0xAD; header[67] = 0xDE
    header[68] = 0x01
    struct.pack_into('<I', header, 72, wsize)
    struct.pack_into('<I', header, 80, 128)

    fp16_data = np.array(weights_f32, dtype=np.float16).tobytes()
    return bytes(header) + fp16_data


class ANEKernel:
    """Handle to a compiled ANE kernel."""
    def __init__(self, handle, bridge):
        self._handle = handle
        self._bridge = bridge

    def __del__(self):
        if self._handle and self._bridge and self._bridge._lib:
            self._bridge._lib.ane_bridge_free(self._handle)
            self._handle = None


class ANEBridge:
    """
    Bridge to Apple Neural Engine via native Objective-C code.

    If native/build/libane_bridge.dylib exists, provides full ANE access:
    compile MIL programs, evaluate kernels, read/write IOSurface tensors.

    Otherwise, provides hardware detection and MIL generation utilities.
    """

    def __init__(self, lib_path=None):
        self._lib = None
        self.native_available = False
        self.ane_available = check_ane_available()
        self.chip, self.brand = detect_chip()
        self.memory_gb = get_memory_gb()
        self.ane_tops = ANE_TOPS.get(self.chip, 0)
        self.gpu_tflops = GPU_TFLOPS.get(self.chip, 0)

        # Try to load native bridge
        if lib_path is None:
            native_dir = Path(__file__).parent / "native" / "build"
            lib_path = native_dir / "libane_bridge.dylib"

        if os.path.exists(lib_path):
            try:
                self._load_native(str(lib_path))
            except Exception as e:
                print(f"Warning: Failed to load native bridge: {e}")

    def _load_native(self, lib_path):
        """Load the native bridge library and set up ctypes bindings."""
        self._lib = ctypes.CDLL(lib_path)

        # ane_bridge_init() -> int
        self._lib.ane_bridge_init.restype = ctypes.c_int
        self._lib.ane_bridge_init.argtypes = []

        # ane_bridge_compile(...) -> ANEKernelHandle*
        self._lib.ane_bridge_compile.restype = ctypes.c_void_p
        self._lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,        # mil_text, mil_len
            ctypes.c_char_p, ctypes.c_size_t,        # weight_data, weight_len
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_inputs, input_sizes
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_outputs, output_sizes
        ]

        # ane_bridge_eval(kernel) -> bool
        self._lib.ane_bridge_eval.restype = ctypes.c_bool
        self._lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]

        # ane_bridge_write_input(kernel, idx, data, bytes)
        self._lib.ane_bridge_write_input.restype = None
        self._lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]

        # ane_bridge_read_output(kernel, idx, data, bytes)
        self._lib.ane_bridge_read_output.restype = None
        self._lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]

        # ane_bridge_free(kernel)
        self._lib.ane_bridge_free.restype = None
        self._lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

        # ane_bridge_get_compile_count() -> int
        self._lib.ane_bridge_get_compile_count.restype = ctypes.c_int
        self._lib.ane_bridge_get_compile_count.argtypes = []

        # ane_bridge_build_weight_blob(src, rows, cols, out_len) -> uint8_t*
        self._lib.ane_bridge_build_weight_blob.restype = ctypes.c_void_p
        self._lib.ane_bridge_build_weight_blob.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)
        ]

        # ane_bridge_free_blob(ptr)
        self._lib.ane_bridge_free_blob.restype = None
        self._lib.ane_bridge_free_blob.argtypes = [ctypes.c_void_p]

        # Initialize
        ret = self._lib.ane_bridge_init()
        if ret == 0:
            self.native_available = True
        else:
            print("Warning: ane_bridge_init() failed — ANE not accessible")

    def get_info(self):
        """Return hardware information."""
        return {
            "chip": self.chip,
            "brand": self.brand,
            "ane_available": self.ane_available,
            "native_available": self.native_available,
            "ane_tops": self.ane_tops,
            "gpu_tflops": self.gpu_tflops,
            "memory_gb": self.memory_gb,
            "compile_count": self.get_compile_count() if self.native_available else 0,
        }

    def get_compile_count(self):
        """Get number of ANE compilations (for exec() restart budgeting)."""
        if not self.native_available:
            return 0
        return self._lib.ane_bridge_get_compile_count()

    def compile_kernel(self, mil_text, weight_data=None,
                       input_sizes=None, output_sizes=None):
        """
        Compile a MIL program into an ANE kernel.

        Args:
            mil_text: MIL program text (str or bytes)
            weight_data: Raw weight blob bytes (None for weightless kernels)
            input_sizes: List of input tensor sizes in bytes
            output_sizes: List of output tensor sizes in bytes

        Returns:
            ANEKernel handle (or None on failure)
        """
        if not self.native_available:
            raise RuntimeError("Native bridge not available. Build with: cd native && make bridge")

        if isinstance(mil_text, str):
            mil_text = mil_text.encode('utf-8')

        n_in = len(input_sizes) if input_sizes else 1
        n_out = len(output_sizes) if output_sizes else 1

        in_sizes = (ctypes.c_size_t * n_in)(*input_sizes) if input_sizes else (ctypes.c_size_t * 1)(0)
        out_sizes = (ctypes.c_size_t * n_out)(*output_sizes) if output_sizes else (ctypes.c_size_t * 1)(0)

        w_ptr = None
        w_len = 0
        if weight_data:
            w_ptr = ctypes.c_char_p(weight_data)
            w_len = len(weight_data)

        handle = self._lib.ane_bridge_compile(
            mil_text, len(mil_text),
            w_ptr, w_len,
            n_in, in_sizes,
            n_out, out_sizes
        )

        if not handle:
            return None
        return ANEKernel(handle, self)

    def eval(self, kernel):
        """Evaluate (run) a compiled kernel on ANE."""
        if not self.native_available:
            raise RuntimeError("Native bridge not available")
        return self._lib.ane_bridge_eval(kernel._handle)

    def write_input(self, kernel, idx, data):
        """Write data to kernel input IOSurface."""
        if not self.native_available:
            raise RuntimeError("Native bridge not available")
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        self._lib.ane_bridge_write_input(kernel._handle, idx, data, len(data))

    def read_output(self, kernel, idx, nbytes, dtype=np.float32):
        """Read data from kernel output IOSurface."""
        if not self.native_available:
            raise RuntimeError("Native bridge not available")
        buf = (ctypes.c_uint8 * nbytes)()
        self._lib.ane_bridge_read_output(kernel._handle, idx, buf, nbytes)
        return np.frombuffer(bytes(buf), dtype=dtype)

    def free_kernel(self, kernel):
        """Free a compiled kernel."""
        if kernel and kernel._handle and self._lib:
            self._lib.ane_bridge_free(kernel._handle)
            kernel._handle = None

    # -----------------------------------------------------------------------
    # MIL generation utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def gen_dynamic_matmul_mil(ic, oc, seq):
        """
        Generate MIL for dynamic matmul: y = x @ W where both come from input.

        This is the KEY BREAKTHROUGH from maderix/ANE: weights are packed
        into the input IOSurface alongside activations, so kernels are
        compiled ONCE and weights can be updated every step without
        recompilation.

        Input layout: [1, IC, 1, SEQ+OC] fp16
          - sp[0:SEQ]      = activations x[IC, SEQ]
          - sp[SEQ:SEQ+OC] = weight W[IC, OC] (each channel d holds W[d, :])
        Output: [1, OC, 1, SEQ] fp16
        """
        sp_total = seq + oc
        mil = f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp_total}]> x) {{
        string to16 = const()[name = string("to16"), val = string("fp16")];
        tensor<fp16, [1, {ic}, 1, {sp_total}]> xh = cast(dtype = to16, x = x)[name = string("cin")];
        tensor<int32, [4]> ba = const()[name = string("ba"), val = tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> sa = const()[name = string("sa"), val = tensor<int32, [4]>([1,{ic},1,{seq}])];
        tensor<fp16, [1,{ic},1,{seq}]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string("act")];
        tensor<int32, [4]> bw = const()[name = string("bw"), val = tensor<int32, [4]>([0,0,0,{seq}])];
        tensor<int32, [4]> sw = const()[name = string("sw"), val = tensor<int32, [4]>([1,{ic},1,{oc}])];
        tensor<fp16, [1,{ic},1,{oc}]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string("wt")];
        tensor<int32, [4]> ra = const()[name = string("ra"), val = tensor<int32, [4]>([1,1,{ic},{seq}])];
        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra,x=act)[name=string("a2")];
        tensor<int32, [4]> pm = const()[name = string("pm"), val = tensor<int32, [4]>([0,1,3,2])];
        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string("a3")];
        tensor<int32, [4]> rw = const()[name = string("rw"), val = tensor<int32, [4]>([1,1,{ic},{oc}])];
        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wt)[name=string("W")];
        bool bF = const()[name = string("bF"), val = bool(false)];
        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string("mm")];
        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string("yt")];
        tensor<int32, [4]> ro = const()[name = string("ro"), val = tensor<int32, [4]>([1,{oc},1,{seq}])];
        tensor<fp16, [1,{oc},1,{seq}]> yr = reshape(shape=ro,x=yt)[name=string("yr")];
        string to32 = const()[name = string("to32"), val = string("fp32")];
        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype = to32, x = yr)[name = string("cout")];
    }} -> (y);
}}"""
        return mil

    @staticmethod
    def gen_conv_mil(ic, oc, sp):
        """
        Generate MIL for a convolution (used for static-weight matmul).

        ANE implements matmul as 1x1 convolution internally.
        Input: [1, IC, 1, SP] fp32 -> cast fp16 -> conv -> cast fp32
        Weight: [OC, IC, 1, 1] fp16 (embedded as constant BLOBFILE)
        """
        return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{
        string pt = const()[name = string("pt"), val = string("valid")];
        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
        int32 gr = const()[name = string("gr"), val = int32(1)];
        string to16 = const()[name = string("to16"), val = string("fp16")];
        tensor<fp16, [1, {ic}, 1, {sp}]> xh = cast(dtype = to16, x = x)[name = string("cast_in")];
        tensor<fp16, [{oc}, {ic}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{oc}, {ic}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {oc}, 1, {sp}]> yh = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = xh)[name = string("conv")];
        string to32 = const()[name = string("to32"), val = string("fp32")];
        tensor<fp32, [1, {oc}, 1, {sp}]> y = cast(dtype = to32, x = yh)[name = string("cast_out")];
    }} -> (y);
}}"""

    # -----------------------------------------------------------------------
    # Build helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def build_native(target="bridge"):
        """Build native code. Requires macOS + Xcode CLI tools."""
        if sys.platform != "darwin":
            print("ERROR: Can only build on macOS")
            return False

        native_dir = Path(__file__).parent / "native"
        if not native_dir.exists():
            print(f"ERROR: {native_dir} not found")
            return False

        result = subprocess.run(
            ["make", target],
            cwd=str(native_dir),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}")
            return False
        print(f"Build successful: {target}")
        return True

    @staticmethod
    def run_probe(probe_name):
        """Build and run a probe/benchmark tool."""
        if sys.platform != "darwin":
            print("ERROR: Probes require macOS on Apple Silicon")
            return None

        native_dir = Path(__file__).parent / "native"
        exe = native_dir / "build" / probe_name

        if not exe.exists():
            print(f"Building {probe_name}...")
            result = subprocess.run(
                ["make", f"build/{probe_name}"],
                cwd=str(native_dir),
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Build failed:\n{result.stderr}")
                return None

        result = subprocess.run(
            [str(exe)],
            capture_output=True, text=True,
            timeout=120
        )
        return result.stdout


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANE Bridge — Apple Neural Engine direct access")
    parser.add_argument("--build", choices=["bridge", "probes", "all", "train"],
                       help="Build native code")
    parser.add_argument("--probe", choices=["sram_bench", "sram_probe", "api_exploration",
                                            "inmem_basic", "inmem_bench", "inmem_peak",
                                            "test_weight_reload", "test_dynamic_matmul"],
                       help="Run a probe/benchmark")
    parser.add_argument("--info", action="store_true", help="Show hardware info")
    parser.add_argument("--test-matmul", action="store_true",
                       help="Test dynamic matmul MIL generation")
    args = parser.parse_args()

    if args.build:
        ANEBridge.build_native(args.build)
    elif args.probe:
        output = ANEBridge.run_probe(args.probe)
        if output:
            print(output)
    elif args.test_matmul:
        mil = ANEBridge.gen_dynamic_matmul_mil(768, 768, 256)
        print(mil)
    else:
        # Default: show info
        bridge = ANEBridge()
        info = bridge.get_info()
        print("=== Apple Neural Engine Bridge ===\n")
        for k, v in info.items():
            print(f"  {k}: {v}")

        if bridge.native_available:
            print("\n  Native bridge: LOADED")
            print(f"  Compile count: {bridge.get_compile_count()}")
        else:
            print("\n  Native bridge: NOT BUILT")
            print("  Build with: cd native && make bridge")
            print("  Or: python ane_bridge.py --build bridge")

        print(f"\n  ANE peak: {info['ane_tops']} TOPS")
        print(f"  GPU peak: {info['gpu_tflops']} TFLOPS")
        print(f"  Memory:   {info['memory_gb']:.0f} GB unified")

        if not info['ane_available']:
            print("\n  ANE not available on this system.")
            print("  Requires Apple Silicon (M1/M2/M3/M4) running macOS 15+.")
