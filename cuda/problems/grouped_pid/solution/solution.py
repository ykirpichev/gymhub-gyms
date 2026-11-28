def solve(num_pid_m, num_pid_n, group_m):
    out = []
    for pid in range(num_pid_m * num_pid_n):
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_m
        group_size_m = min(num_pid_m - first_pid_m, group_m)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        out.append([pid_m, pid_n])
    return out
