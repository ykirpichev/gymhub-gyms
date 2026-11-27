# LRU cache

**Level:** medium   **Teaches:** class design, hash map plus doubly linked list, O(1) updates

## Task

Write a class `LRUCache` that stores up to `capacity` key/value pairs. When it is full and a new key arrives, it throws out the key that was used least recently. Both `get` and `put` should take O(1) time.

This exercise has no `solve`. The tests build `LRUCache(capacity)`, replay `ops` against it in order, and compare the list of values returned by every `get`.

## Arguments

| Name       | Type         | Role                                                      |
| ---------- | ------------ | --------------------------------------------------------- |
| `capacity` | `int`        | maximum number of keys, `>= 1`                            |
| `ops`      | `list[list]` | `["get", key]` or `["put", key, value]`, applied in order |
| returns    | `list[int]`  | what each `get` returned, in order (built by the tests)   |

Your class:

| Method                 | Behavior                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| `__init__(capacity)`   | empty cache                                                              |
| `get(key)`             | value for `key`, or `-1` if absent; a hit makes `key` the most recent    |
| `put(key, value)`      | insert or overwrite; makes `key` the most recent; evicts the least recent key if the cache now holds more than `capacity` keys |

Solve it in Python or C++. In C++ write the class with these members:

```cpp
class LRUCache {
  public:
    LRUCache(int capacity);
    int get(int key);
    void put(int key, int value);
};
```

In C++ keys are always `int`. The few tests that use string keys map each string to a distinct int before calling your class.

## Rules

- "Used" means a successful `get` or any `put` of that key.
- Overwriting an existing key never evicts anything.
- Keys are hashable (ints or strings). Values are non-negative ints, so `-1` is never a stored value, but `0` is.

## Example

```
capacity = 2
put(1, 10)  put(2, 20)          cache: 1, 2        (oldest first)
get(1) -> 10                    cache: 2, 1
put(3, 30)                      evicts 2 -> 1, 3
get(2) -> -1
put(4, 40)                      evicts 1 -> 3, 4
get(1) -> -1   get(3) -> 30   get(4) -> 40
results = [10, -1, -1, 30, 40]
```

## How to think about it

A dict finds a key in O(1) but does not know the usage order. A list knows the order but moving or removing an item from the middle costs O(n). The classic answer combines a dict from key to node with a doubly linked list of nodes ordered by recency: unlink a node and relink it at the head in O(1), evict from the tail in O(1). Sentinel head and tail nodes remove most edge cases. (`collections.OrderedDict` with `move_to_end` and `popitem(last=False)` is the same structure in the standard library; using it is allowed, but try writing the linked list once.)

## Tests

Empty op lists, gets on an empty cache, capacity 1, overwrites, recency refresh on `get`, a stored value of `0`, and random op streams. Benchmark input: capacity 50,000 with 300,000 random ops. `bench` fails if the median is over **300 ms** in Python or **50 ms** in C++; keeping the order in a plain list or vector (O(capacity) per operation) takes far longer in either language.
