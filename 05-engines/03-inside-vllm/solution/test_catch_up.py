import catch_up


def test_a_long_prompt_catches_up_while_decoding_continues():
    reqs = [{"id": "chat-A", "computed": 5, "total": 6},
            {"id": "chat-B", "computed": 9, "total": 10},
            {"id": "contract", "computed": 0, "total": 20}]
    plan = catch_up.step(reqs)
    # both chats get their one token first; the contract takes the rest of the 8
    assert plan == {"chat-A": 1, "chat-B": 1, "contract": 6}
    assert sum(plan.values()) <= catch_up.BUDGET


def test_nothing_is_scheduled_for_a_request_that_has_caught_up():
    assert catch_up.step([{"id": "x", "computed": 4, "total": 4}]) == {}
