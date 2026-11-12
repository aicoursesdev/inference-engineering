from diffusion_cost import image_seconds, intensity, per_step_flops, tokens


def test_token_counts():
    assert tokens(1024) == 64 * 64 + 512
    assert tokens(512) == 32 * 32 + 512


def test_one_step_is_compute_bound_on_every_chip():
    assert round(per_step_flops(1024) / 1e12) == 111
    assert intensity(1024) > 1000  # far above any chip's ridge point


def test_h100_best_case():
    assert round(image_seconds("H100-SXM"), 1) == 5.6


def test_price_floor_follows_the_price_table():
    from diffusion_cost import usd_per_image
    assert 0.003 < usd_per_image("H100-SXM") < 0.007
