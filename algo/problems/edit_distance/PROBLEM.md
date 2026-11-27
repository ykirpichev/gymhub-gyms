# Edit distance

**Level:** medium   **Teaches:** 2D dynamic programming, rolling rows, iterative over recursive

## Task

Find the smallest number of single-character edits that turn string `a` into string `b`. An edit inserts one character, deletes one character, or replaces one character with another. Each edit costs 1. This is the Levenshtein distance.

## Arguments

| Name    | Type  | Role                     |
| ------- | ----- | ------------------------ |
| `a`     | `str` | source string, may be empty |
| `b`     | `str` | target string, may be empty |
| returns | `int` | minimum number of edits  |

Solve it in Python or C++. In C++ the function is:

```cpp
int solve(string& a, string& b);
```

## Formula

Let `D[i][j]` be the distance between the first `i` characters of `a` and the first `j` characters of `b`.

```
D[i][0] = i                      (delete all i characters)
D[0][j] = j                      (insert all j characters)
D[i][j] = D[i-1][j-1]                                    if a[i-1] == b[j-1]
        = 1 + min(D[i-1][j-1],   replace
                  D[i-1][j],     delete a[i-1]
                  D[i][j-1])     insert b[j-1]           otherwise
answer  = D[len(a)][len(b)]
```

## Example

```
a = "sunday"   b = "saturday"
s(a)(t)u(n->r)day: insert 'a', insert 't', replace 'n' with 'r'
result = 3
```

## How to think about it

Row `i` of the table depends only on row `i-1` and on the cell to its left, so you never need the full `len(a) × len(b)` table: keep two rows (or one row plus a saved diagonal value). That cuts memory from O(n·m) to O(min(n, m)) and is also faster in Python, since it allocates far fewer lists. A memoized recursion computes the same thing but pays for a function call and a cache entry per cell and can recurse `len(a) + len(b)` levels deep.

## Tests

Both empty, one side empty, identical strings, no shared letters, repeated letters, and random strings up to 400 characters over small alphabets. Benchmark input: a random 2,000-character string and a copy with 400 random edits. `bench` fails if the median is over **1000 ms** in Python or **20 ms** in C++. A clean O(n·m) loop with a rolling row runs in about 300 ms in Python and about 5 ms in C++ on the reference machine; a memoized recursion in C++ takes about 30 ms.
