"""pycudabrot — a Mandelbrot renderer for your terminal, JIT-compiled to the GPU.

The CUDA C kernel below is compiled at first use by CuPy (via NVRTC), so the
package itself is pure Python and ships as a ``noarch`` conda package.
"""

import cupy as cp

_KERNEL_SOURCE = r"""
extern "C" __global__
void mandelbrot(unsigned char *rgb, int width, int height,
                double center_re, double center_im, double step, int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double c_re = center_re + (x - width / 2.0) * step;
    double c_im = center_im - (y - height / 2.0) * step;

    double re = 0.0, im = 0.0;
    int it = 0;
    while (re * re + im * im <= 4.0 && it < max_iter) {
        double tmp = re * re - im * im + c_re;
        im = 2.0 * re * im + c_im;
        re = tmp;
        ++it;
    }

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (it < max_iter) {
        // Smooth (fractional) escape count for band-free coloring.
        float log_zn = logf((float)(re * re + im * im)) * 0.5f;
        float t = (it + 1.0f - log2f(log_zn)) / max_iter;
        r = 0.5f + 0.5f * cosf(3.0f + 12.0f * t);
        g = 0.5f + 0.5f * cosf(3.6f + 12.0f * t);
        b = 0.5f + 0.5f * cosf(4.2f + 12.0f * t);
    }
    int i = 3 * (y * width + x);
    rgb[i + 0] = (unsigned char)(255.0f * r);
    rgb[i + 1] = (unsigned char)(255.0f * g);
    rgb[i + 2] = (unsigned char)(255.0f * b);
}
"""

_mandelbrot = cp.RawKernel(_KERNEL_SOURCE, "mandelbrot")


def render(width, height, center=(-0.6, 0.0), span=3.2, max_iter=256):
    """Render the Mandelbrot set on the GPU, returning an (H, W, 3) uint8 array."""
    rgb = cp.empty((height, width, 3), dtype=cp.uint8)
    block = (16, 16)
    grid = (-(-width // block[0]), -(-height // block[1]))
    _mandelbrot(
        grid,
        block,
        (
            rgb,
            cp.int32(width),
            cp.int32(height),
            cp.float64(center[0]),
            cp.float64(center[1]),
            cp.float64(span / width),
            cp.int32(max_iter),
        ),
    )
    return cp.asnumpy(rgb)
