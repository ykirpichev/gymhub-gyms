## Hint 1

In single mode there are `grid * block` threads; elements beyond them are missed.

## Hint 2

In stride mode the owner of element `j` is `j % (grid * block)`.
