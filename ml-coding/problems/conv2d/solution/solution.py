import numpy as np


def solve(x, weight, bias, stride, padding):
    image = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(weight, dtype=np.float64)
    offset = np.asarray(bias, dtype=np.float64)
    if image.ndim != 3 or kernel.ndim != 4 or offset.ndim != 1:
        raise ValueError("x is (C_in, H, W), weight is (C_out, C_in, kH, kW), bias is (C_out,)")
    channels, height, width = image.shape
    out_channels, kernel_channels, k_h, k_w = kernel.shape
    if kernel_channels != channels or offset.shape != (out_channels,) or k_h < 1 or k_w < 1:
        raise ValueError("channel counts and bias length must match the weight")
    if not _is_int(stride) or not _is_int(padding) or stride < 1 or padding < 0:
        raise ValueError("stride must be an integer >= 1 and padding an integer >= 0")
    out_h = (height + 2 * padding - k_h) // stride + 1
    out_w = (width + 2 * padding - k_w) // stride + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("kernel does not fit in the padded image")

    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))
    output = np.empty((out_channels, out_h, out_w), dtype=np.float64)
    for channel in range(out_channels):
        for i in range(out_h):
            for j in range(out_w):
                top, left = i * stride, j * stride
                window = padded[:, top:top + k_h, left:left + k_w]
                # Cross-correlation: the kernel is used as stored, not rotated.
                output[channel, i, j] = np.sum(window * kernel[channel]) + offset[channel]
    return output


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
