def solve(requests, max_batch, mode):
    if mode not in ("continuous", "static") or max_batch < 1:
        raise ValueError("bad mode or batch size")
    waiting = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    running = {}
    finish = [None] * len(requests)
    t = 0
    while waiting or running:
        if not running and waiting and requests[waiting[0]][0] > t:
            t = requests[waiting[0]][0]
        if mode == "continuous" or not running:
            while waiting and len(running) < max_batch and requests[waiting[0]][0] <= t:
                running[waiting.pop(0)] = 0
        for index in list(running):
            running[index] += 1
            if running[index] == requests[index][1]:
                finish[index] = t + 1
                del running[index]
        t += 1
    return finish
