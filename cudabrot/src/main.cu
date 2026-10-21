// cudabrot — a Mandelbrot renderer for your terminal, computed on the GPU.
//
// Every pixel is one CUDA thread. The result is drawn with Unicode
// half-blocks (▀) and 24-bit ANSI colors, so each terminal cell holds
// two vertically stacked pixels.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(err_));             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// A smooth cosine palette (courtesy of the Inigo Quilez school of shading).
__device__ uchar3 palette(float t)
{
    float r = 0.5f + 0.5f * cosf(3.0f + 12.0f * t);
    float g = 0.5f + 0.5f * cosf(3.6f + 12.0f * t);
    float b = 0.5f + 0.5f * cosf(4.2f + 12.0f * t);
    return make_uchar3(static_cast<unsigned char>(255.0f * r),
                       static_cast<unsigned char>(255.0f * g),
                       static_cast<unsigned char>(255.0f * b));
}

__global__ void mandelbrot(uchar3 *out, int width, int height, double center_re,
                           double center_im, double step, int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

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

    uchar3 color = make_uchar3(0, 0, 0);
    if (it < max_iter) {
        // Smooth (fractional) escape count for band-free coloring.
        float log_zn = logf(static_cast<float>(re * re + im * im)) * 0.5f;
        float nu = it + 1.0f - log2f(log_zn);
        color = palette(nu / max_iter);
    }
    out[y * width + x] = color;
}

static void terminal_size(int *width, int *height)
{
    int cols = 0, rows = 0;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    }
#else
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        cols = ws.ws_col;
        rows = ws.ws_row;
    }
#endif
    if (cols > 0 && rows > 2) {
        *width = cols;
        *height = 2 * (rows - 3);  // half-blocks: two pixels per row
    } else {
        *width = 100;
        *height = 56;
    }
}

#ifdef _WIN32
// The Windows console needs to be told to emit UTF-8 (for ▀) and to
// interpret ANSI escape sequences (for the 24-bit colors).
static void enable_ansi_console()
{
    SetConsoleOutputCP(CP_UTF8);
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(out, &mode)) {
        SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}
#endif

int main(int argc, char **argv)
{
    // Usage: cudabrot [center_re center_im span [max_iter]]
    double center_re = -0.6, center_im = 0.0, span = 3.2;
    int max_iter = 256;
    if (argc >= 4) {
        center_re = std::atof(argv[1]);
        center_im = std::atof(argv[2]);
        span = std::atof(argv[3]);
    }
    if (argc >= 5) {
        max_iter = std::atoi(argv[4]);
    }

#ifdef _WIN32
    enable_ansi_console();
#endif

    int width = 0, height = 0;
    terminal_size(&width, &height);
    double step = span / width;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    uchar3 *d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, sizeof(uchar3) * width * height));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mandelbrot<<<grid, block>>>(d_pixels, width, height, center_re, center_im, step, max_iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<uchar3> pixels(static_cast<size_t>(width) * height);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, sizeof(uchar3) * pixels.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));

    // Draw two pixel rows per terminal row: the upper pixel is the foreground
    // of a ▀ half-block, the lower pixel is its background.
    std::string frame;
    frame.reserve(static_cast<size_t>(width) * height * 20);
    char buf[64];
    for (int y = 0; y + 1 < height; y += 2) {
        for (int x = 0; x < width; ++x) {
            uchar3 top = pixels[static_cast<size_t>(y) * width + x];
            uchar3 bottom = pixels[static_cast<size_t>(y + 1) * width + x];
            std::snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm",
                          top.x, top.y, top.z, bottom.x, bottom.y, bottom.z);
            frame += buf;
            frame += "▀";
        }
        frame += "\x1b[0m\n";
    }
    std::fputs(frame.c_str(), stdout);

    std::printf("%s | %dx%d px | %d iterations | kernel: %.2f ms\n", prop.name, width,
                height, max_iter, ms);
    return 0;
}
