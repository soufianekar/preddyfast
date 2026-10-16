"""Pure query-param parsing."""

from utils import scorecard
from utils.url_state import query_choice


def test_obs_mode_from_the_query_falls_back_to_raw():
    def obs(params, variable="temperature_2m"):
        return query_choice(params, "obs", scorecard.obs_modes(variable), "raw")

    assert obs({}) == "raw"
    assert obs({"obs": "representativeness"}) == "representativeness"
    assert obs({"obs": "height_corrected"}) == "height_corrected"
    assert obs({"obs": "height_corrected"}, "wind_speed_10m") == "raw"
    assert obs({"obs": "HEIGHT"}) == "raw"
