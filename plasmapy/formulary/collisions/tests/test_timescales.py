"""Test functionality of timescales in `plasmapy.formulary.collisions.timescales`."""
import numpy as np
import pytest

from astropy import units as u

from plasmapy.formulary.collisions.timescales import Hellinger
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import InvalidParticleError


class TestTimescales:
    version_2009 = 2009
    args_2009 = {
        "T": 8 * u.K,
        "n_i": 4.0e5 * u.m**-3,
        "ions": [Particle("He+"), Particle("H+")],
        "par_speeds": [30 * u.m / u.s, 50 * u.m / u.s],
    }

    version_2010 = 2010
    args_2010 = {
        "T_par": 100 * u.K,
        "T_perp": 200 * u.K,
        "n_i": 4.06e10 * u.m**-3,
        "ions": [Particle("He+"), Particle("H+")],
        "par_speeds": [30 * u.m / u.s, 50 * u.m / u.s],
    }

    version_2016 = 2016
    args_2016 = {
        "T_par": [100 * u.K, 500 * u.K],
        "T_perp": [600 * u.K, 900 * u.K],
        "n_i": 5000 * u.m**-3,
        "ions": [Particle("He+"), Particle("H+")],
        "par_speeds": [600, 700] * u.m / u.s,
        "perp_speeds": [800, 900] * u.m / u.s,
    }

    _kwargs_2009 = {"version": version_2009, "kwargs": args_2009}
    _kwargs_2010 = {"version": version_2010, "kwargs": args_2010}
    _kwargs_2016 = {"version": version_2016, "kwargs": args_2016}

    @pytest.mark.parametrize(
        "version, kwargs, _error",
        [
            (
                "wrong type",
                _kwargs_2009["kwargs"],
                TypeError,
            ),
            (
                2000,
                _kwargs_2009["kwargs"],
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "T": "wrong type"},
                TypeError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "T": [8e-9, 8.5e-9] * u.K},
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "T": -1 * u.K},
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "n_i": "wrong type"},
                TypeError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "n_i": 6 * u.m / u.s},
                u.UnitTypeError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "n_i": [4, 2, 3] * u.m**-3},
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "n_i": np.ones((2, 2)) * u.m**-3},
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "ions": {"not": "a particle"}},
                InvalidParticleError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "par_speeds": [10, 40, 60] * u.m / u.s},
                ValueError,
            ),
            (
                _kwargs_2009["version"],
                {**_kwargs_2009["kwargs"], "par_speeds": [10, 40] * u.rad / u.s},
                u.UnitTypeError,
            ),
            # 2010
            (
                "wrong type",
                _kwargs_2010["kwargs"],
                TypeError,
            ),
            (
                2000,
                _kwargs_2010["kwargs"],
                ValueError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_par": "wrong type"},
                TypeError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_par": [8e-9, 8.5e-9] * u.K},
                ValueError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_par": -1 * u.K},
                ValueError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_perp": "wrong type"},
                TypeError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_perp": [8e-9, 8.5e-9] * u.K},
                ValueError,
            ),
            (
                _kwargs_2010["version"],
                {**_kwargs_2010["kwargs"], "T_perp": -1 * u.K},
                ValueError,
            ),
        ],
    )
    def test_raises(self, version, kwargs, _error):
        with pytest.raises(_error):
            Hellinger(kwargs, version)
