# pycudabrot 🐍⚡

The Python sibling of [`cudabrot`](../cudabrot): the same GPU Mandelbrot
kernel, but embedded as a string and JIT-compiled at runtime by
[CuPy](https://cupy.dev) — packaged as a **noarch** conda package with
Pixi's `pixi-build-python` backend.

No compiler needed at build time; the CUDA magic happens on the user's
machine via NVRTC.

## Run it

```console
pixi run render
pixi run seahorse
```

## Package it

```console
pixi publish --target-dir ./dist
```

This produces a `noarch` `pycudabrot-0.1.0-*.conda` package.
