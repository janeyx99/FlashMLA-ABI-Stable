# FlashMLA (SGLang fork)

A fork of [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA), packaged as a standalone wheel for SGLang.

It differs from upstream in three ways:

- **Upstream's current kernels are carried verbatim.** `csrc/` is a byte-for-byte copy of upstream's NVIDIA sources, so syncing is a directory replacement rather than a merge. The Huawei Ascend NPU target is not carried.
- **The kernels upstream dropped are kept.** Upstream's 2026.09.30 release removed Hopper (SM90), dense MLA decoding, and the V3.2 / V3.2-no-RoPE / V4 KV cache formats. SGLang still serves all of them, so they live on in `csrc_legacy/` as a frozen snapshot, built as a second extension with its own op namespace. See [csrc_legacy/README.md](csrc_legacy/README.md).
- **One Python surface for both.** Every entry point takes `use_legacy=False`; passing `True` routes the same call to the frozen kernels. There is no separate compatibility module.

Upstream's documentation of the kernels themselves — KV cache formats, the `indices` tensor, the fused norm-RoPE-attn-RoPE-cast kernel and its weight permutation — still applies and is not duplicated here.

## Requirements

- CUDA 13.1 and above. CUDA 13.2+ is recommended: it provides the native `cvt.rn.bf16x2` conversions the quantized decoding kernels use, and without them they fall back to a slower widening through FP32.
- PyTorch 2.11 and above with a CUDA 13 runtime for CUDA wheels.
- An SM100 / SM103 GPU for the current kernels; the frozen kernels additionally cover SM90.

Wheels are built against PyTorch's stable C++ ABI and CPython's Limited API (`abi3`, CPython 3.10+). The C++ ABI target is 2.10; CUDA wheels require PyTorch 2.11+ built for CUDA 13 because they link `libcudart.so.13`. The same wheel can run on later CUDA 13 Torch releases, subject to the GPU architectures compiled into it.

## Installation

```bash
git clone https://github.com/sgl-project/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v . --no-build-isolation
```

`--no-build-isolation` is required: `setup.py` imports `torch` while building, and this repository does not declare it as a PEP 518 build requirement.

Install `ninja` first, and make sure it is on `PATH` as an executable. Without it PyTorch falls back to the distutils backend, which ignores `MAX_JOBS` and compiles a couple of translation units at a time — minutes become hours, and the only sign is one line of warning early in the log.

Build-time switches:

| variable | default | effect |
|---|---|---|
| `FLASH_MLA_BUILD_LEGACY` | `1` | Build the frozen `csrc_legacy/` extension. `0` ships only the current kernels; `use_legacy=True` then raises. |
| `FLASH_MLA_BUILD_DENSE_PREFILL` | `0` | Build the CUTLASS dense FMHA prefill/backward. It is half the extension's object code for an op SGLang never calls, so it is off by default; the ops stay registered and raise if called. |
| `FLASH_MLA_SKIP_REG_SPILL_CHECK` | unset | Skip the post-build register-spill check. A debugging escape hatch; released wheels are built with the check on. |
| `NVCC_THREADS` | `32` | `nvcc --threads`. |

After the build, a register-spill check runs over the produced `.so`. CUTLASS kernels are exempt; if any kernel defined by this repository spills, the build fails.

## Tests

```bash
python tests/test-sparse-prefill.py
python tests/test-sparse-decode.py
python tests/test-fused-norm-rope-attn-rope-cast.py
```

The fused-kernel test additionally needs [TileLang](https://github.com/tile-ai/tilelang), [Tile-Kernels](https://github.com/deepseek-ai/TileKernels/) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). Tests for the frozen kernels live in [tests/legacy/](tests/legacy/).

## Acknowledgement

FlashMLA is DeepSeek's work, inspired by the [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [CUTLASS](https://github.com/nvidia/cutlass) projects.

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li and Shengyu Liu and Yuanhang Sun},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
