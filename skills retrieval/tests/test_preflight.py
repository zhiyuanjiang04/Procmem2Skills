from skills_retrieval.preflight import count_tokens_approx, will_fit


def test_approx_token_count_monotonic():
    a = count_tokens_approx("hello")
    b = count_tokens_approx("hello world this is longer")
    assert b > a


def test_will_fit_true_for_short():
    assert will_fit("short prompt", model_context_limit=1000, safety_margin=100)


def test_will_fit_false_for_long():
    long_text = "x " * 2000
    assert not will_fit(long_text, model_context_limit=100, safety_margin=10)
