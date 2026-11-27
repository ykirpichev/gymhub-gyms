# Timing GPU code

```python
start = time.time()
c = a @ b              # a, b: 8192 x 8192 CUDA tensors
elapsed = time.time() - start
```

`elapsed` is 30 µs, far less than the matmul can take. Why?
