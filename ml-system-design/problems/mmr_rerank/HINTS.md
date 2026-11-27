## Hint 1

Keep `closest[i]`: the max similarity from `i` to anything chosen so far, starting at 0. After each pick `p`, update `closest[i] = max(closest[i], similarity[i][p])`.

## Hint 2

Scan candidates in index order and replace the best only on a strictly larger score; that gives the lowest index on ties.
