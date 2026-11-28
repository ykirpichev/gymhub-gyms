# Hillis–Steele scan

Implement the Hillis–Steele inclusive scan the way a GPU runs it, and report the work done. `xs` is a list of numbers.
For `d = 1, 2, 4, …` while `d < len(xs)`: every element `i ≥ d` becomes `x[i] + x[i - d]`, all at the same time (from
the values before the step). Count one addition per updated element.

Return `[scan, steps, additions]`.

Example: `[3, 1, 7, 0, 4, 1, 6, 3]` → scan `[3, 4, 11, 11, 15, 16, 22, 25]`, 3 steps, 7 + 6 + 4 = 17 additions (a
sequential scan needs only 7).
