## Hint 1

`arrivals[i + max_batch - 1]` is the time the batch would be full, if that request exists.

## Hint 2

Once `start` is known, take requests while `arrivals[j] <= start` and fewer than `max_batch` are taken; then advance `i` past them.
