def solve(taken, then_cost, else_cost):
    issued = useful = 0
    for w in range(0, len(taken), 32):
        lanes = taken[w:w + 32]
        on = sum(1 for t in lanes if t)
        if on:
            issued += then_cost
            useful += on * then_cost
        if on < len(lanes):
            issued += else_cost
            useful += (len(lanes) - on) * else_cost
    return [issued, useful]
