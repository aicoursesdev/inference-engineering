from policy import Keys, TokenBucket, digest


def test_bucket_refills_with_the_clock():
    now = [0.0]
    b = TokenBucket(rate=10, burst=100, clock=lambda: now[0])
    assert b.take(100) == 0          # a full bucket allows a burst
    assert b.take(50) == 5.0         # empty: 50 tokens at 10/s is 5 s away
    now[0] = 5.0
    assert b.take(50) == 0


def test_keys_are_matched_by_hash():
    info = {"customer": "tidewater", "tokens_per_min": 600}
    keys = Keys({digest("sk-tw"): info})
    assert keys.check("Bearer sk-tw") == info
    assert keys.check("Bearer sk-other") is None
    assert keys.check(None) is None
    assert keys.charge(info, 600) == 0
    assert keys.charge(info, 60) > 0
