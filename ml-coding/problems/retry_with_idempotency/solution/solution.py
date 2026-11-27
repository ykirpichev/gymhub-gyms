class IdempotentExecutor:
    def __init__(self, sleep, base, cap, max_attempts):
        self.sleep = sleep
        self.base = base
        self.cap = cap
        self.max_attempts = max_attempts
        self.stored = {}

    def run(self, key, fn, transient):
        if key in self.stored:
            return self.stored[key]
        delays = []
        for attempt in range(self.max_attempts):
            try:
                value = fn()
            except transient as exc:
                if attempt + 1 == self.max_attempts:
                    result = self._failure(exc, attempt + 1, delays)
                    self.stored[key] = result
                    return result
                delay = min(self.cap, self.base * 2 ** attempt)
                delays.append(delay)
                self.sleep(delay)
            except Exception as exc:
                result = self._failure(exc, attempt + 1, delays)
                self.stored[key] = result
                return result
            else:
                result = {"ok": True, "value": value, "attempts": attempt + 1, "delays": delays}
                self.stored[key] = result
                return result

    @staticmethod
    def _failure(exc, attempts, delays):
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": attempts,
            "delays": delays,
        }
