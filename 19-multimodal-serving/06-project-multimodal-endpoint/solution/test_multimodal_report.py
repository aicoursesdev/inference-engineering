import os

os.environ.setdefault("EMBED_GGUF", "unused")
from multimodal_report import predicted_prompt_tokens, usd, voice_turn_ms


def test_price_of_a_second_on_the_cheapest_l4_row():
    # data/prices.json: RunPod L4 community cloud, $0.44 per hour
    assert round(usd(3600), 2) == 0.44


def test_a_two_second_picture_question_costs_a_fraction_of_a_cent():
    assert 0.0002 < usd(2.0) < 0.0003


def test_predictions_match_what_the_worked_run_reported():
    # server prompt_tokens in the lesson's table, for each picture
    for (w, h), reported in {(448, 448): 284, (1280, 720): 1224,
                             (1920, 1080): 2719}.items():
        assert predicted_prompt_tokens(w, h) == reported


def test_voice_turn_adds_the_three_stages():
    assert round(voice_turn_ms(0.037, 0.090)) == 427     # the lesson's line
    assert round(voice_turn_ms(0.1, 0.2, 0.3)) == 600
