# Reverse a vector

**Level:** easy   **Teaches:** index mapping, coalesced reads and writes

## Task

Copy `A` into `B` in reverse order. The first element of `B` is the last element of `A`. `A` is not changed.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
B[i] = A[N - 1 - i]    for i = 0 .. N-1
```

## Example

```
A = [1, 2, 3, 4, 5]
B = [5, 4, 3, 2, 1]
```

## How to think about it

Every output depends on exactly one input, so give each thread one index `i`. Thread `i` reads `A[N - 1 - i]` and writes `B[i]`.

Check the memory pattern. Neighbouring threads in a warp write neighbouring addresses of `B`, and they read neighbouring addresses of `A`, just in descending order. The hardware still combines those reads into a few wide transactions. So the simple version already runs close to the copy speed of the GPU. The rest is about moving more bytes per instruction.

## In an interview

A small indexing puzzle that hides a memory-access question: reading forward and writing backward, are both coalesced?

Follow-ups to answer out loud once your kernel passes:

- Do it in place with half the threads. What must each thread do, and is there a race?
- Reverse within blocks through shared memory. When does that beat the direct version?
- What happens at the middle element when `n` is odd?
