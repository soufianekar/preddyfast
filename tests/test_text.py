from utils.text import in_sentence


def test_units_and_acronyms_keep_their_case():
    assert in_sentence("Geopotential height (500 hPa)") == "geopotential height (500 hPa)"
    assert in_sentence("500 hPa geopotential height") == "500 hPa geopotential height"
    assert in_sentence("Max Temperature") == "max temperature"
    assert in_sentence("Mean sea level pressure") == "mean sea level pressure"
    assert in_sentence("MSLP and CAPE") == "MSLP and CAPE"
