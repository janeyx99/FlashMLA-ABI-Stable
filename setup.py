import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

from setup_legacy_sources import LEGACY_CUDA_ARCHS, legacy_sources

# The frozen pre-2026.09.30 kernels build as their own extension: their own op namespace and
# their own architectures, but the same stable C++ ABI as the current kernels, so that the wheel
# is abi3 as a whole. FLASH_MLA_BUILD_LEGACY=0 ships only the current kernels.
# See csrc_legacy/README.md.
OUR_OP_SOURCES = ["csrc_ops/sched_meta.cpp"]
EXT_NAME_LEGACY = "flash_mla._C_legacy"
BUILD_LEGACY = os.environ.get("FLASH_MLA_BUILD_LEGACY", "1") == "1"

# The CUTLASS FMHA dense prefill is upstream's, and it is half the CUDA extension's object code
# (29 MB of 61 MB) for an op SGLang never calls -- it is not a DeepSeek MLA kernel at all, just a
# dense attention one that happens to live here. It is left out by default; a build that needs
# `dense_prefill_fwd` / `dense_prefill_bwd` sets FLASH_MLA_BUILD_DENSE_PREFILL=1. The schemas
# still come from upstream's csrc/api/api.cpp, so the ops exist but have no CUDA kernel
# registered, and calling one raises rather than failing the load.
BUILD_DENSE_PREFILL = os.environ.get("FLASH_MLA_BUILD_DENSE_PREFILL", "0") == "1"

THIS_DIR = Path(__file__).resolve().parent
VERSION_PATH = Path("flash_mla") / "VERSION"

# Stable dispatcher API shared by both build platforms. The fused kernels live
# in their own translation unit.
API_SOURCES = [
    "csrc/api/api.cpp",
    "csrc/api/sparse_prefill.cpp",
    "csrc/api/sparse_decode.cpp",
    "csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp",
]

DENSE_PREFILL_SOURCES = [
    "csrc/api/dense_fwd.cpp",
    "csrc/api/dense_bwd.cpp",
    # sm100 dense prefill (cutlass FMHA): forward and backward
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",
]

# The extension modules, one per build platform
EXT_NAME_CUDA = "flash_mla.cuda"

# `-gencode` targets. We use the architecture-specific ones (sm_100a / sm_103a) instead of the
# family-specific one (sm_100f) because the kernels use `cvt` with `.scaled::n2::ue8m0`, and
# sm_103a additionally enables better code generation for `exp`.
CUDA_ARCHS = ["sm_100a", "sm_103a"]

# Build against PyTorch's stable C++ ABI.
STABLE_ABI_FLAGS = [
    "-DTORCH_TARGET_VERSION=0x020a000000000000",  # PyTorch >= 2.10
]
CUDA_STABLE_ABI_FLAGS = [*STABLE_ABI_FLAGS, "-DUSE_CUDA"]

# Head counts (H_Q) the fused core_attn kernel is instantiated for: 128 (2-CTA cluster) and 64
HEADS_FOR_FUSED_CORE_ATTN = (64, 128)

CUDA_KERNEL_SOURCES = [
    # Small kernels for decoding: tile scheduler metadata generation and the split-KV combine
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",

    # sm100 sparse prefill
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",

    # sm100 sparse prefill for small topk, and head128 decoding
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu",

    # sm100 sparse decode (head 64, with and without split-KV, on the fp8 and the fp4 cache)
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu",

    # Fused Norm + RoPE + Core Attn + RoPE + Cast: 6 instantiations per head count
    # (V4.1 fp8 prefill, V4.1 fp8 decode and V4.1 fp4 decode, each with and without q norm)
    *[f"csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/{model}_h{h}_{mode}_{norm}.cu"
      for h in HEADS_FOR_FUSED_CORE_ATTN
      for model, mode in (("v41", "prefill"), ("v41", "decode"), ("v41fp4", "decode"))
      for norm in ("norm", "nonorm")],
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu",
]
def get_cuda_arch_flags(archs=None):
    arch_flags = []
    for arch in (CUDA_ARCHS if archs is None else archs):
        compute_arch = arch.removeprefix("sm_")
        arch_flags.extend(["-gencode", f"arch=compute_{compute_arch},code={arch}"])
    return arch_flags


# Exempt from the spill check: upstream's split-KV h64 sparse decode really does spill --
# ptxas reports 20 bytes spill stores / 28 bytes spill loads in a 16-byte stack frame, on CUDA
# 13.2/13.3/13.4 alike. Unchanged from upstream 0f11e17, whose own build fails its own check
# here. Verified correct on sm_103a hardware, so this is a perf cost upstream ships, not a bug.
KNOWN_UPSTREAM_FRAME_KERNELS = ("flash_fwd_splitkv_mla_fp8_sparse_kernel",)


def is_known_upstream_frame(mangled_name: str) -> bool:
    return any(k in mangled_name for k in KNOWN_UPSTREAM_FRAME_KERNELS)


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def build_on_cuda_platform():
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
    )

    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable.
    # For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")
    assert major > 13 or (major == 13 and minor >= 1), \
        "Compiling the kernels requires NVCC 13.1 or higher (cvt with .scaled::n2::ue8m0)."

    nvcc_archs = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--list-gpu-arch"], text=True
    )
    if "compute_107" in nvcc_archs:
        CUDA_ARCHS.append("sm_107a")
        LEGACY_CUDA_ARCHS.append("sm_107a")

    class BuildExtensionWithPostActions(BuildExtension):
        """
        Run the register-spill check after building.
        """
        def run(self):
            super().run()

            if not self.dry_run:
                from tests.kernelkit import check_kernel_reg_spill_in_artifact

                so_paths = [
                    os.path.join(self.build_lib, self.get_ext_filename(EXT_NAME_CUDA)),
                    self.get_ext_fullpath(EXT_NAME_CUDA),    # May point to the extension in the source tree for editable installs
                ]
                so_paths = list(set(so_paths))  # deduplicate
                assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
                for so_path in so_paths:
                    spilled_kernels = check_kernel_reg_spill_in_artifact(
                        so_path, quiet=True, suppress_checking_env_var="FLASH_MLA_SKIP_REG_SPILL_CHECK"
                    )
                    # The restored upstream CUTLASS FMHA kernels (cutlass::device_kernel<...>) keep
                    # their stack frames, so they are exempt. Only the kernels defined by this
                    # repository have to be spill-free.
                    own_spilled_kernels = [
                        name for name in spilled_kernels
                        if not name.startswith("_ZN7cutlass") and not is_known_upstream_frame(name)
                    ]
                    num_exempt_kernels = len(spilled_kernels) - len(own_spilled_kernels)
                    if num_exempt_kernels > 0:
                        print(f"{num_exempt_kernels} upstream kernel(s) keep their stack frames (exempt)")
                    if len(own_spilled_kernels) > 0:
                        for kernel_name in own_spilled_kernels:
                            print(f"  {kernel_name}")
                        print("Register spilling detected. Build failed!")
                        sys.exit(1)

    ext_modules = [
        CUDAExtension(
            name=EXT_NAME_CUDA,
            sources=[
                *API_SOURCES,
                *(DENSE_PREFILL_SOURCES if BUILD_DENSE_PREFILL else []),
                *CUDA_KERNEL_SOURCES,
                # Ours, not upstream's: an op upstream does not register. It lives outside
                # csrc/ so that directory can be replaced wholesale on a sync.
                *OUR_OP_SOURCES,
            ],
            extra_compile_args={
                # The host-side (`.cpp`) translation units are not compiled by nvcc, so `__CUDACC__`
                # is not defined and neither kerutils' nor our own platform switch is set automatically.
                "cxx": [
                    *CUDA_STABLE_ABI_FLAGS,
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Wno-deprecated-declarations",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                    "-DKERUTILS_IS_BUILD_ON_CUDA",
                ],
                "nvcc": get_nvcc_thread_args() + CUDA_STABLE_ABI_FLAGS + [
                    "-O3",
                    "-std=c++20",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                    # Shrinks the fatbin; nvcc 13.0+. Upstream's -lineinfo / --source-in-ptx
                    # are dropped with it: together they take the wheel from 21 MB to 8 MB
                    # and neither changes codegen (resource usage is identical either way).
                    "--compress-mode=size",
                    "--diag-suppress=445",  # "constant "N" is not used in declaring the parameter types of function template"
                    "-DNDEBUG",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                ] + get_cuda_arch_flags(),
            },
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc_ops",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "tools" / "util" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
                Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",    # for cuda/std headers in CUDA 13+
                Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
            ],
            # On a machine without the NVIDIA driver (e.g. a CPU-only build node) `-lcuda` needs
            # the stub libraries that ship with the toolkit
            extra_link_args=[
                f"-L{Path(CUDA_HOME) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
                f"-L{Path(CUDA_HOME) / 'targets' / 'sbsa-linux' / 'lib' / 'stubs'}",
                "-lcuda",
            ],
            # Registration uses STABLE_TORCH_LIBRARY and does not need the
            # Python C API, so one extension can serve every CPython >= 3.10.
            py_limited_api=True,
        ),
    ]

    if BUILD_LEGACY:
        # The frozen tree is built against the stable ABI too. A wheel is only as portable as its
        # least portable extension, so one CPython-locked extension here would undo the abi3 build
        # of the current kernels above, whatever the wheel tag says.
        ext_modules.append(CUDAExtension(
            name=EXT_NAME_LEGACY,
            sources=legacy_sources(),
            extra_compile_args={
                "cxx": [
                    *CUDA_STABLE_ABI_FLAGS,
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Wno-deprecated-declarations",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                    "-DKERUTILS_IS_BUILD_ON_CUDA",
                    # Export only the module entry point and the op registration: the two
                    # extensions are loaded into one process and must not see each other.
                    "-fvisibility=hidden",
                    "-fvisibility-inlines-hidden",
                ],
                "nvcc": get_nvcc_thread_args() + CUDA_STABLE_ABI_FLAGS + [
                    "-O3",
                    "-std=c++20",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--compress-mode=size",
                    "--diag-suppress=445",
                    "-DNDEBUG",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                    "-Xcompiler", "-fvisibility=hidden",
                ] + get_cuda_arch_flags(LEGACY_CUDA_ARCHS),
            },
            include_dirs=[
                THIS_DIR / "csrc_legacy",
                # The vendored dependencies are shared. The legacy tree pins none of its own, so
                # an upstream cutlass bump still reaches it; give csrc_legacy its own 3rdparty/
                # and change these lines if that ever matters.
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "tools" / "util" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
                Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",
                Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
            ],
            extra_link_args=[
                f"-L{Path(CUDA_HOME) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
                f"-L{Path(CUDA_HOME) / 'targets' / 'sbsa-linux' / 'lib' / 'stubs'}",
                "-lcuda",
            ],
            # Registration is STABLE_TORCH_LIBRARY here too, so this extension needs no more of
            # the Python C API than the current one does.
            py_limited_api=True,
        ))

    return ext_modules, BuildExtensionWithPostActions



def get_version() -> str:
    # One source of truth, read by flash_mla.__version__ too. The release workflow
    # overwrites it; it carries no local version segment, because PyPI rejects those
    # and the wheel index spells the CUDA build as +cuNNN itself.
    return (THIS_DIR / VERSION_PATH).read_text().strip()


def main():
    # CUDA only; this fork does not carry upstream's Ascend NPU target.
    ext_modules, build_ext = build_on_cuda_platform()
    print(f"Building {len(ext_modules)} extension(s): {[e.name for e in ext_modules]}")

    setup(
        # Distribution name; the import name stays `flash_mla`, so that downstream
        # code reads the same either way. Mirrors sgl-deep-gemm / deep_gemm.
        name="sgl-flash-mla",
        version=get_version(),
        description="FlashMLA attention kernels for SGLang",
        long_description=(THIS_DIR / "README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/sgl-project/FlashMLA",
        license="MIT",
        license_files=["LICENSE"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Environment :: GPU :: NVIDIA CUDA",
        ],
        python_requires=">=3.10",
        # The extensions are built against PyTorch's stable C++ ABI and hold no
        # symbols from libtorch's C++ one, so they keep working as torch moves
        # forward. The C++ ABI floor is 2.10 (TORCH_TARGET_VERSION), while
        # published CUDA 13 wheels require a CUDA 13 Torch runtime, available
        # from 2.11 onward. Keep the package floor independent of SGLang's
        # current serving-runtime pin; a CUDA 12 Torch wheel cannot supply
        # libcudart.so.13, and pip cannot express that CUDA constraint.
        install_requires=["torch>=2.11"],
        packages=find_packages(include=["flash_mla"]),
        package_data={"flash_mla": [VERSION_PATH.name]},
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )


if __name__ == "__main__":
    main()
