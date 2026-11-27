# Hints for LRU cache

## Hint 1

You need two things at once: find a key fast, and know which key is the oldest. A dict gives the first. For the second, think of a queue where any element can be pulled out of the middle and moved to the front cheaply.

## Hint 2

Use a doubly linked list of nodes `(key, value, prev, next)` with two sentinel nodes, `head` and `tail`, so the list is never empty. The dict maps `key -> node`. Write two helpers: `_unlink(node)` and `_push_front(node)`.

## Hint 3

`get`: look up the node, unlink it, push it to the front, return its value. `put`: if the key exists, update the value and move it to the front; otherwise create a node, push it to the front, add it to the dict, and if the dict is over capacity unlink `tail.prev` and delete its key from the dict. Storing the key in the node is what makes that last delete possible. `__slots__` on the node class makes it noticeably faster.

In C++, a `std::list<pair<int, int>>` plus `unordered_map<int, list<pair<int, int>>::iterator>` is the same design. `list::splice(list.begin(), list, it)` moves a node to the front in O(1) without invalidating iterators.
