"""The gateway's rules, with no web framework in them so they can be tested (lesson 10.4)."""

import hashlib
import hmac
import time


class TokenBucket:
    """Holds up to `burst` tokens and refills `rate` per second.
    take(n) spends n if it can; otherwise it says how long to wait."""

    def __init__(self, rate, burst, clock=time.monotonic):
        self.rate, self.burst, self.clock = rate, burst, clock
        self.level, self.last = burst, clock()

    def take(self, n=1):
        now = self.clock()
        self.level = min(self.burst, self.level + (now - self.last) * self.rate)
        self.last = now
        if self.level >= n:
            self.level -= n
            return 0.0
        return (n - self.level) / self.rate  # seconds until it would fit


def digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


class Keys:
    """API keys, stored only as hashes, each with its customer and token budget."""

    def __init__(self, table):
        # table: {sha256-of-key: {"customer": ..., "tokens_per_min": ...}}
        self.table = table
        self.buckets = {}

    def check(self, header):
        """The customer for an `Authorization: Bearer <key>` header, or None."""
        if not header or not header.startswith("Bearer "):
            return None
        d = digest(header[7:])
        for known, info in self.table.items():
            if hmac.compare_digest(known, d):
                return info
        return None

    def charge(self, info, tokens):
        """Seconds to wait before this customer may spend `tokens` more; 0 means go."""
        per_s = info["tokens_per_min"] / 60
        bucket = self.buckets.setdefault(
            info["customer"], TokenBucket(per_s, info["tokens_per_min"]))
        return bucket.take(tokens)
