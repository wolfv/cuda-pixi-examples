# cudabrot 🌀⚡

A Mandelbrot renderer for your terminal, computed on your GPU — and a
complete example of building **and packaging** CUDA software with
[Pixi](https://pixi.sh) and the `pixi-build-cmake` backend.

Every pixel is one CUDA thread. The image is drawn with Unicode half-blocks
and 24-bit ANSI colors, so each terminal cell holds two pixels.

## Requirements

- Linux (x86-64) with an NVIDIA GPU and a driver supporting CUDA 12
- [Pixi](https://pixi.sh/latest/#installation)

That's it. No system CUDA toolkit installation — `nvcc`, the CUDA runtime,
compilers and CMake all come from conda-forge, managed by Pixi.

## Run it

```console
pixi run render     # the classic full-set view
pixi run seahorse   # deep zoom into seahorse valley
```

The first invocation builds the package from source (Pixi downloads the
toolchain, compiles, and caches the result). Subsequent runs are instant
unless sources changed.

## Package it

```console
pixi publish --target-dir ./dist
```

This produces a relocatable `cudabrot-0.1.0-<build>.conda` package that you
can upload to any conda channel (e.g. your own channel on
[prefix.dev](https://prefix.dev)).

## No GPU? Solving still works

To resolve/build on a machine without an NVIDIA driver (e.g. CI), override
the detected virtual package:

```console
CONDA_OVERRIDE_CUDA=12 pixi install
```
