"""The sun guard decides whether an OPEN roof may be reported as safe.

Every unknown has to resolve to "not safe": an ASCOM client parks equipment on
this answer, so a failed calculation must never read as daylight-safe.
"""

CONFIG = {'latitude': '40.0', 'longitude': '-74.0', 'sun_angle_threshold': '-17.0'}


def test_sun_below_threshold_is_safe(app):
    app.calculate_sun_angle = lambda config=None: -30.0
    assert app.is_sun_safe_for_open(CONFIG) is True


def test_sun_above_threshold_is_not_safe(app):
    app.calculate_sun_angle = lambda config=None: 10.0
    assert app.is_sun_safe_for_open(CONFIG) is False


def test_sun_exactly_at_threshold_is_not_safe(app):
    app.calculate_sun_angle = lambda config=None: -17.0
    assert app.is_sun_safe_for_open(CONFIG) is False


def test_unknown_sun_angle_is_not_safe(app):
    """calculate_sun_angle returning None used to be 0.0 - a fabricated reading."""
    app.calculate_sun_angle = lambda config=None: None
    assert app.is_sun_safe_for_open(CONFIG) is False


def test_unparseable_threshold_is_not_safe(app):
    app.calculate_sun_angle = lambda config=None: -30.0
    assert app.is_sun_safe_for_open({**CONFIG, 'sun_angle_threshold': 'dark'}) is False


def test_missing_threshold_is_not_safe(app):
    app.calculate_sun_angle = lambda config=None: -30.0
    assert app.is_sun_safe_for_open({'latitude': '40.0', 'longitude': '-74.0'}) is False


def test_failing_sun_calculation_is_not_safe(app):
    def boom(config=None):
        raise RuntimeError("ephem exploded")

    app.calculate_sun_angle = boom
    assert app.is_sun_safe_for_open(CONFIG) is False


def test_calculate_sun_angle_returns_none_on_bad_coordinates(app):
    """Unknown must be distinguishable from a real horizon reading of 0.0."""
    assert app.calculate_sun_angle(
        {'latitude': 'north-ish', 'longitude': '-74.0', 'sun_angle_threshold': '-17.0'}
    ) is None


def test_calculate_sun_angle_returns_a_real_angle(app):
    angle = app.calculate_sun_angle(CONFIG)
    assert angle is not None
    assert -90.0 <= angle <= 90.0
