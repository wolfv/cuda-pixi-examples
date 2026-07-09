# CUDA + Pixi examples

Companion code for the blog post *"Fast, reproducible CUDA development —
and packaging — with Pixi"*.

Two flavors of the same idea (a GPU-rendered Mandelbrot set in your
terminal), packaged with two different Pixi build backends:

| Directory | Language | Backend | Package type |
|---|---|---|---|
| [`cudabrot/`](./cudabrot) | C++ / CUDA | `pixi-build-cmake` | native `linux-64` binary |
| [`pycudabrot/`](./pycudabrot) | Python + CuPy | `pixi-build-python` | `noarch` python |

## Quick start

Requirements: Linux x86-64 / Windows x86-64, an NVIDIA GPU with a CUDA 12 or later driver, and
[Pixi](https://pixi.sh/latest/#installation). Everything else — `nvcc`,
CUDA runtime, compilers, CMake, Python — is pulled from conda-forge
automatically.

```console
cd cudabrot     # or pycudabrot
pixi run render
```
