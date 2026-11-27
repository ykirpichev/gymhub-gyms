# Hints for Edit distance

## Hint 1

Think about the last characters. If `a[-1] == b[-1]`, they can be matched for free and the answer is the distance between the two shorter prefixes. Otherwise the last edit was a replace, a delete from `a`, or an insert from `b`: try all three and keep the cheapest.

## Hint 2

Turn that into a table `D[i][j]` over prefix lengths, filled row by row. The first row is `0, 1, 2, ..., len(b)` and the first column is `0, 1, ..., len(a)`. Fill it with two nested loops instead of recursion.

## Hint 3

Keep only the previous row `prev` and the current row `cur`. Swap so the inner loop runs over the shorter string. In the inner loop, avoid calling `min` on a three-element tuple; two `if` comparisons on local variables are noticeably faster in CPython.

In C++, one `vector<int>` row plus a variable holding the old diagonal value is enough. Avoid allocating a new row inside the outer loop.
