# GYM_STARTER


class IdempotentExecutor:
    """Retry transient failures with capped exponential backoff and cache by idempotency key."""

    def __init__(self, sleep, base, cap, max_attempts):
        pass

    def run(self, key, fn, transient):
        """Run fn, or return the stored result when key was already completed."""
        pass
