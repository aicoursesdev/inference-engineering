import radix_demo


def test_the_second_step_skips_the_system_prompt():
    tree = {}
    first = radix_demo.SYSTEM + [9001, 9002]
    assert radix_demo.reused(tree, first) == 0
    second = first + [7001] * 400
    assert radix_demo.reused(tree, second) == len(first)


def test_a_new_task_reuses_only_the_shared_system_prompt():
    tree = {}
    radix_demo.reused(tree, radix_demo.SYSTEM + [9001])
    other = radix_demo.SYSTEM + [4242] * 50
    assert radix_demo.reused(tree, other) == 3000
