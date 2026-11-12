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
        raise NotImplementedError("TODO 10.4: refill by the time passed (capped at burst), spend n if it fits, else return the seconds to wait, see the lesson's 'Limit by tokens, not requests' section")


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
        raise NotImplementedError("TODO 10.4: take the key from 'Bearer <key>', hash it and compare with hmac.compare_digest; None if unknown, see the lesson's 'Know who is calling' section")

    def charge(self, info, tokens):
        """Seconds to wait before this customer may spend `tokens` more; 0 means go."""
        raise NotImplementedError("TODO 10.4: one TokenBucket per customer (rate tokens_per_min / 60, burst tokens_per_min); return its take(tokens), see the lesson's 'Limit by tokens, not requests' section")
