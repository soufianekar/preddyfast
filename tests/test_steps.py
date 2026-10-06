import numpy as np

from utils.earthmover_api import _describe_steps


def test_ifs_steps():
    lead = np.r_[np.arange(0, 145, 3), np.arange(150, 361, 6)].astype(float)
    assert _describe_steps(lead) == "3 h to T+144, 6 h after"


def test_aifs_steps():
    assert _describe_steps(np.arange(0, 361, 6.0)) == "6 h"
