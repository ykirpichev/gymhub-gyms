# Worked example: vector add

**Level:** easy   **Teaches:** checking the toolchain, the shape of a `solve` function

## Task

This exercise is already solved. It computes `C = A + B` in CUDA, PyTorch, and Triton, and exists to check that all three run on this machine.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `const float*` | `N`   | input  |
| `C`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
C[i] = A[i] + B[i]    for i in [0, N)
```

## Example

```
A = [1, 2, 3]
B = [4, 5, 6]
C = [5, 7, 9]
```

## How to think about it

Run `gpu-gym test playground --lang all`. If all three languages pass, the setup works. Then read `solutions/playground/solution.cu`: it shows the `solve` signature, the launch, and the bounds check. `vector_add` is the same task with the kernel body left empty.

`solve` receives GPU pointers (or tensors) and writes into `C`. It does not return anything.
