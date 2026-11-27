# __TITLE__

**Level:** easy   **Teaches:** describe the one or two ideas this exercise practices

The scaffold starts as a vector add so the runner works before you change anything. Rewrite this statement, then change `problem.py` (the reference and the cases) and the `solve` signature in `starter/` together. `gpu-gym show __SLUG__` adds the Tests and Files sections from `problem.py`. Don't write them here.

## Task

One to three plain sentences: what goes in, what comes out.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `const float*` | `N`   | input  |
| `C`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
C[i] = A[i] + B[i]
```

## Example

```
A = [1, 2, 3]
B = [4, 5, 6]
C = [5, 7, 9]
```

## How to think about it

What is independent, what has to be shared, and where the parallelism is. Describe the naive version and why it is slow. Don't include code.
