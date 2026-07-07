"""Terminal front-end: draws the GPU-rendered image with ANSI half-blocks."""

import shutil
import sys
import time

import cupy as cp

from pycudabrot import render


def main() -> None:
    # Usage: pycudabrot [center_re center_im span [max_iter]]
    center, span, max_iter = (-0.6, 0.0), 3.2, 256
    if len(sys.argv) >= 4:
        center = (float(sys.argv[1]), float(sys.argv[2]))
        span = float(sys.argv[3])
    if len(sys.argv) >= 5:
        max_iter = int(sys.argv[4])

    cols, rows = shutil.get_terminal_size(fallback=(100, 31))
    width, height = cols, 2 * (rows - 3)

    start = time.perf_counter()
    rgb = render(width, height, center=center, span=span, max_iter=max_iter)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Two pixel rows per terminal row: the upper pixel is the foreground of
    # a ▀ half-block, the lower pixel is its background.
    lines = []
    for y in range(0, height - 1, 2):
        cells = []
        for x in range(width):
            tr, tg, tb = rgb[y, x]
            br, bg, bb = rgb[y + 1, x]
            cells.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
        lines.append("".join(cells) + "\x1b[0m")
    print("\n".join(lines))

    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{gpu} | {width}x{height} px | {max_iter} iterations | {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
