def solve(width, height, bx, by, block_x, block_y, thread_x, thread_y):
    col = block_x * bx + thread_x
    row = block_y * by + thread_y
    if row >= height or col >= width:
        return -1
    return row * width + col
