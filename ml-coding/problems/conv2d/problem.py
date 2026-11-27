"""Grader for conv2d. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def reference(x, weight, bias, stride, padding):
    image = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(weight, dtype=np.float64)
    offset = np.asarray(bias, dtype=np.float64)
    if image.ndim != 3 or kernel.ndim != 4 or offset.ndim != 1:
        return "ValueError"
    channels, height, width = image.shape
    out_channels, kernel_channels, k_h, k_w = kernel.shape
    if kernel_channels != channels or offset.shape != (out_channels,) or k_h < 1 or k_w < 1:
        return "ValueError"
    if not _is_int(stride) or not _is_int(padding) or stride < 1 or padding < 0:
        return "ValueError"
    out_h = (height + 2 * padding - k_h) // stride + 1
    out_w = (width + 2 * padding - k_w) // stride + 1
    if out_h < 1 or out_w < 1:
        return "ValueError"
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))
    output = np.empty((out_channels, out_h, out_w), dtype=np.float64)
    for channel in range(out_channels):
        for i in range(out_h):
            for j in range(out_w):
                top, left = i * stride, j * stride
                window = padded[:, top:top + k_h, left:left + k_w]
                output[channel, i, j] = np.sum(window * kernel[channel]) + offset[channel]
    return output


def call(module, args):
    """Bad shapes, stride, padding or a kernel that does not fit must raise ValueError."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError(
                "expected ValueError: stride >= 1, padding >= 0, matching channels, "
                "a bias entry per output channel, and a positive output size"
                if want == "ValueError" else "raised ValueError, but this convolution is valid")
        return
    try:
        got_array = np.asarray(got, dtype=np.float64)
    except (ValueError, TypeError):
        raise AssertionError(f"expected an array of shape {list(want.shape)}, got {type(got).__name__}") from None
    if got_array.shape != want.shape:
        raise AssertionError(
            f"output shape {list(got_array.shape)} != {list(want.shape)}. "
            "H_out = (H + 2*padding - kH) // stride + 1, and the same for W "
            "(floor division; a kernel that does not fit raises ValueError).")
    flipped_weight = np.asarray(case["weight"], dtype=np.float64)[:, :, ::-1, ::-1]
    flipped = reference(case["x"], flipped_weight, case["bias"], case["stride"], case["padding"])
    if isinstance(flipped, np.ndarray) and flipped.shape == want.shape:
        matches_flip = np.allclose(got_array, flipped, atol=1e-8, rtol=1e-6)
        matches = np.allclose(got_array, want, atol=1e-8, rtol=1e-6)
        if matches_flip and not matches:
            raise AssertionError(
                "output matches the kernel rotated 180 degrees. "
                "This operation is cross-correlation, which is what deep-learning conv2d computes: "
                "do not flip the filter.")
    default_compare(got, want, 1e-8, 1e-6, "output")


def _random(seed, channels, height, width, out_channels, k_h, k_w, stride, padding, name):
    rng = np.random.default_rng(seed)
    return {
        "_name": name,
        "x": rng.normal(size=(channels, height, width)).tolist(),
        "weight": rng.normal(size=(out_channels, channels, k_h, k_w)).tolist(),
        "bias": rng.normal(size=out_channels).tolist(),
        "stride": stride,
        "padding": padding,
    }


def functional_cases():
    image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    kernel = [[[[1.0, 2.0], [3.0, 4.0]]]]
    return [
        {"_name": "example", "x": image, "weight": kernel, "bias": [0.5], "stride": 1, "padding": 0},
        {"_name": "stride 2 and padding 1", "x": image, "weight": kernel, "bias": [0.5], "stride": 2, "padding": 1},
        {"_name": "output shape uses floor division",
         "x": np.arange(1, 21, dtype=float).reshape(1, 5, 4).tolist(),
         "weight": [[[[1.0, 2.0], [0.0, -1.0]]]], "bias": [0.0], "stride": 2, "padding": 0},
        {"_name": "two input and two output channels",
         "x": [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]]],
         "weight": [
             [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]],
             [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]],
         ],
         "bias": [0.5, -0.5], "stride": 1, "padding": 0},
        {"_name": "non-square kernel",
         "x": [np.arange(1, 16, dtype=float).reshape(3, 5).tolist()],
         "weight": [[[[1.0, 0.0, -1.0], [0.5, 0.0, 0.25]]]], "bias": [0.25], "stride": 1, "padding": 0},
        {"_name": "asymmetric kernel, not a flipped convolution",
         "x": [[[1.0, 2.0, 3.0, 4.0]]], "weight": [[[[1.0, 0.0, -1.0]]]], "bias": [0.0],
         "stride": 1, "padding": 0},
        {"_name": "padding makes the output larger",
         "x": [[[1.0, 2.0], [3.0, 4.0]]], "weight": [[[[1.0]]]], "bias": [0.0], "stride": 1, "padding": 1},
        {"_name": "zero kernel, bias only",
         "x": [[[1.0, 2.0], [3.0, 4.0]]], "weight": [[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]],
         "bias": [3.0, -2.0], "stride": 1, "padding": 0},
        _random(11, 3, 9, 8, 4, 3, 2, 2, 1, "random 3x9x8, stride 2"),
        _random(12, 8, 16, 16, 6, 3, 3, 1, 1, "random 8x16x16"),
        {"_name": "stride 0", "x": image, "weight": kernel, "bias": [0.5], "stride": 0, "padding": 0},
        {"_name": "negative padding", "x": image, "weight": kernel, "bias": [0.5], "stride": 1, "padding": -1},
        {"_name": "kernel does not fit", "x": [[[1.0, 2.0], [3.0, 4.0]]],
         "weight": [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]], "bias": [0.0],
         "stride": 1, "padding": 0},
        {"_name": "channel mismatch", "x": image, "weight": np.ones((1, 2, 2, 2)).tolist(),
         "bias": [0.0], "stride": 1, "padding": 0},
        {"_name": "bias length mismatch", "x": image, "weight": kernel, "bias": [0.5, -0.5],
         "stride": 1, "padding": 0},
    ]


def build() -> Problem:
    return Problem(
        slug="conv2d",
        title="2D convolution",
        difficulty="medium",
        summary="Cross-correlation of a multi-channel image with stride, zero padding and bias.",
        params=["x", "weight", "bias", "stride", "padding"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "convolution", "computer vision"],
    )
