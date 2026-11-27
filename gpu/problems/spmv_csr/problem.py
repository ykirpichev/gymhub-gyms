import torch

from gpu_gym.inputs import cuda_empty, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(row_ptr, col_idx, values, x, y, rows, cols, nnz):
    ptr = row_ptr.long()
    lengths = ptr[1:] - ptr[:-1]
    row_of = torch.repeat_interleave(torch.arange(rows, device=x.device), lengths)
    products = values.double() * x.double()[col_idx.long()]
    out = torch.zeros(rows, dtype=torch.float64, device=x.device)
    out.index_add_(0, row_of, products)
    y.copy_(out.float())


def _csr(rows, cols, mean_len, seed, power_law=False):
    """A random CSR matrix. power_law gives a few very long rows and many short or empty ones."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if power_law:
        # Lengths with a heavy tail: about a third of the rows are empty, a few have over 10,000 entries.
        u = torch.rand(rows, generator=generator, dtype=torch.float64)
        lengths = (mean_len / 8.2 * (1.0 / (u + 1e-4) - 1.0)).long()
    else:
        lengths = torch.randint(0, 2 * mean_len + 1, (rows,), generator=generator)
    lengths = lengths.clamp_(0, cols)
    ptr = torch.zeros(rows + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(lengths, 0)
    nnz = int(ptr[-1])
    assert nnz < 1 << 24, "positions must stay exact as float32"
    # Sorted columns inside each row, like a real CSR matrix; repeats are allowed and add up.
    col = torch.randint(0, cols, (nnz,), generator=generator)
    row_of = torch.repeat_interleave(torch.arange(rows), lengths)
    order = torch.argsort(row_of * cols + col)
    col = col[order]
    values = torch.empty(nnz, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    x = torch.empty(cols, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    as_cuda = lambda t: t.to(device="cuda", dtype=torch.float32).contiguous()
    return as_cuda(ptr), as_cuda(col), as_cuda(values), as_cuda(x), nnz


def _case(rows, cols, mean_len, seed, name, power_law=False):
    ptr, col, values, x, nnz = _csr(rows, cols, mean_len, seed, power_law)
    if nnz == 0:
        # The runner needs a non-empty tensor; one explicit zero changes nothing.
        ptr, col, values = ptr.clone(), cuda_tensor([0.0]), cuda_tensor([0.0])
        ptr[1:] = 1.0
        nnz = 1
    return {
        "_name": name,
        "row_ptr": ptr,
        "col_idx": col,
        "values": values,
        "x": x,
        "y": cuda_empty(rows),
        "rows": rows,
        "cols": cols,
        "nnz": nnz,
    }


def functional_cases():
    known = {
        "_name": "example",
        "row_ptr": cuda_tensor([0.0, 2.0, 2.0, 5.0]),
        "col_idx": cuda_tensor([0.0, 3.0, 1.0, 2.0, 3.0]),
        "values": cuda_tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        "x": cuda_tensor([1.0, 10.0, 100.0, 1000.0]),
        "y": cuda_empty(3),
        "rows": 3,
        "cols": 4,
        "nnz": 5,
    }
    cases = [
        _case(1, 1, 1, 1, "1x1"),
        _case(10, 20, 3, 2, "10x20, about 3 per row"),
        _case(1000, 1000, 16, 3, "1000x1000, about 16 per row"),
        _case(5000, 3000, 50, 4, "5000x3000, power-law rows", power_law=True),
        _case(100, 100000, 2000, 5, "100 long rows"),
        _case(100000, 100000, 4, 6, "100000 short rows"),
    ]
    large = _case(1 << 17, 1 << 17, 32, 99, "131072x131072, power-law rows", power_law=True)
    return [known] + cases + [large]


def performance_case(seed: int = 123):
    return _case(1 << 19, 1 << 20, 16, seed, "perf", power_law=True)


def traffic(case):
    rows, cols, nnz = case["rows"], case["cols"], case["nnz"]
    # Row pointers, two arrays of nnz, x and y, each read or written once.
    return ((rows + 1) + 2 * nnz + cols + rows) * 4


def build() -> Problem:
    return Problem(
        slug="spmv_csr",
        title="Sparse matvec (CSR)",
        difficulty="medium",
        summary="Multiply a sparse matrix in CSR format by a dense vector, with rows of very different lengths.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("row_ptr", "tensor"),
            Param("col_idx", "tensor"),
            Param("values", "tensor"),
            Param("x", "tensor"),
            Param("y", "tensor", output=True),
            Param("rows", "int"),
            Param("cols", "int"),
            Param("nnz", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
