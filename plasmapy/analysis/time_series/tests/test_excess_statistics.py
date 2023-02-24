"""Tests for excess_statistics.py"""

import astropy.units as u
import numpy as np
import pytest

from plasmapy.analysis.time_series.excess_statistics import _calculate_pdfs, excess_stat


@pytest.mark.parametrize(
    "signal, thresholds, time_step, pdf, bins, expected",
    [
        (
            [0, 2, 2, 0, 4, 0],
            [1, 3, 5],
            1,
            False,
            32,
            ([3, 1, 0], [2, 1, 0], [1.5, 1, 0], [0.5, 0, 0]),
        ),
        (
            np.array([2, 2, 0, 4, 0]),
            [1, 3, 5],
            1,
            False,
            32,
            ([3, 1, 0], [1, 1, 0], [1.5, 1, 0], [0.5, 0, 0]),
        ),
        (
            [0, 2, 0, 4, 0] * u.eV,
            [1, 3, 5],
            1,
            False,
            32,
            ([2, 1, 0], [2, 1, 0], [1, 1, 0], [0, 0, 0]),
        ),
        (
            [0, 2, 0, 4, 0],
            [1, 3, 5],
            1,
            True,
            2,
            (
                [2, 1, 0],
                [2, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
                [[0, 2], [0, 2], [0, 0]],
                [[0.75, 1.25], [0.75, 1.25], [0, 0]],
            ),
        ),
    ],
)
def test_excess_stat(signal, thresholds, time_step, pdf, bins, expected):
    """test excess_stat function"""
    results = excess_stat(
        signal=signal, thresholds=thresholds, time_step=time_step, pdf=pdf, bins=bins
    )
    assert np.allclose(results[:4], expected[:4])
    if results.hist is not None:
        assert np.allclose(results.hist, expected[4])
        assert np.allclose(results.bin_centers, expected[5])


@pytest.mark.parametrize(
    "signal, thresholds, time_step, pdf, bins",
    [([1, 2], 1, -1, False, 32), ([1, 2], 1, 1, True, 1.5)],
)
def test_excess_stat_exception(signal, thresholds, time_step, pdf, bins):
    """test whether exception is risen"""
    with pytest.raises((ValueError, TypeError)):
        excess_stat(signal, thresholds, time_step, pdf, bins)


@pytest.mark.parametrize(
    "events_per_threshold, bins, expected",
    [
        (
            {1: [1, 1], 3: [1], 5: []},
            2,
            (
                [[0, 2], [0, 2], [0, 0]],
                [[0.75, 1.25], [0.75, 1.25], [0, 0]],
            ),
        )
    ],
)
def test_calculate_pdfs(events_per_threshold, bins, expected):
    """test _calculate_pdfs function"""
    results = _calculate_pdfs(events_per_threshold, bins)
    assert np.allclose(results[0], expected[0])
    assert np.allclose(results[1], expected[1])
