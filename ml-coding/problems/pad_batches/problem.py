"""Grader for pad_batches. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _pad(sequences, pad_id, max_len):
    if max_len is not None and (isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0):
        raise ValueError("max_len must be None or an int >= 0")
    if len(sequences) == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool), []
    rows = []
    for sequence in sequences:
        row = [int(token) for token in sequence]
        if max_len is not None:
            row = row[:max_len]
        rows.append(row)
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), int(pad_id), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=bool)
    lengths = []
    for index, row in enumerate(rows):
        length = len(row)
        lengths.append(length)
        if length:
            ids[index, :length] = row
            mask[index, :length] = True
    return ids, mask, lengths


def reference(sequences, pad_id, max_len):
    try:
        return _pad(sequences, pad_id, max_len)
    except ValueError:
        return "ValueError"


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError for max_len < 0" if want == "ValueError"
                else "raised ValueError on a valid batch"
            )
        return
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError(f"return (ids, mask, lengths), got {type(got).__name__}")
    ids = np.asarray(got[0])
    mask = np.asarray(got[1])
    if ids.dtype != np.dtype(np.int64):
        raise AssertionError(f"ids dtype is {ids.dtype}, expected int64")
    if mask.dtype != np.dtype(bool):
        raise AssertionError(f"mask dtype is {mask.dtype}, expected bool")
    if ids.shape != want[0].shape:
        raise AssertionError(
            f"ids shape {list(ids.shape)} != expected {list(want[0].shape)} "
            "(empty batch is (0, 0); width is the longest sequence after truncation, not max_len)"
        )
    default_compare(ids, want[0], 0, 0, "ids")
    default_compare(mask, want[1], 0, 0, "mask")
    lengths = got[2].tolist() if isinstance(got[2], np.ndarray) else got[2]
    default_compare(lengths, want[2], 0, 0, "lengths")


def _random(seed, batch, max_len, name):
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(batch):
        length = int(rng.integers(0, 12))
        sequences.append(rng.integers(0, 40, size=length).tolist())
    return {
        "_name": name,
        "sequences": sequences,
        "pad_id": int(rng.integers(-2, 3)),
        "max_len": max_len,
    }


def functional_cases():
    return [
        {"_name": "example", "sequences": [[1, 2, 3], [4], [5, 6]], "pad_id": 0, "max_len": None},
        {"_name": "pad id is a real token", "sequences": [[0, 1], [2, 0, 3]], "pad_id": 0, "max_len": None},
        {"_name": "custom pad id", "sequences": [[7], [8, 9]], "pad_id": -1, "max_len": None},
        {"_name": "truncate on the right", "sequences": [[1, 2, 3, 4, 5], [6, 7]], "pad_id": 0, "max_len": 3},
        {"_name": "max_len wider than the batch", "sequences": [[1, 2], [3]], "pad_id": 0, "max_len": 10},
        {"_name": "empty sequence in the batch", "sequences": [[1, 2], [], [3]], "pad_id": 0, "max_len": None},
        {"_name": "all sequences empty", "sequences": [[], []], "pad_id": 0, "max_len": 5},
        {"_name": "empty batch", "sequences": [], "pad_id": 7, "max_len": 4},
        {"_name": "max_len 0", "sequences": [[1, 2, 3], [4]], "pad_id": 0, "max_len": 0},
        {"_name": "single sequence", "sequences": [[8, 9, 10]], "pad_id": 0, "max_len": None},
        {"_name": "one empty sequence", "sequences": [[]], "pad_id": 0, "max_len": None},
        {"_name": "equal lengths", "sequences": [[1, 2], [3, 4]], "pad_id": 0, "max_len": None},
        _random(1, 8, None, "random 8 sequences"),
        _random(2, 15, 6, "random 15 sequences, max_len 6"),
        {"_name": "negative max_len", "sequences": [[1, 2]], "pad_id": 0, "max_len": -1},
    ]


def build() -> Problem:
    return Problem(
        slug="pad_batches",
        title="Pad variable-length token batches",
        difficulty="medium",
        summary="Right-pad token-id sequences into a batch, with a boolean mask, lengths, and optional truncation.",
        params=["sequences", "pad_id", "max_len"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "batching", "sequences"],
    )
