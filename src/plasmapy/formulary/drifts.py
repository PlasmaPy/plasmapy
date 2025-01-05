"""Particle drift velocities."""

__all__ = [
    "diamagnetic_drift",
    "ExB_drift",
    "force_drift",
    "vd_",
    "veb_",
    "vfd_",
]
__aliases__ = ["vd_", "veb_", "vfd_"]

import astropy.units as u
import numpy as np

from plasmapy.utils.decorators import validate_quantities


@validate_quantities
def diamagnetic_drift(
    dp: u.Quantity[u.Pa / u.m],
    B: u.Quantity[u.T],
    n: u.Quantity[u.m**-3],
    q: u.Quantity[u.C],
) -> u.Quantity[u.m / u.s]:
    r"""
    Calculate the diamagnetic fluid perpendicular drift.

    **Aliases:** `vd_`

    Parameters
    ----------
    dp : `~astropy.units.Quantity`
        Pressure gradient vector
    B  : `~astropy.units.Quantity`
        Magnetic field vector
    n  : `~astropy.units.Quantity`
        Number density
    q  : `~astropy.units.Quantity`
        Particle charge

    Returns
    -------
    v: `~astropy.units.Quantity`
        Drift velocity, in m/s

    Notes
    -----
    The diamagnetic drift is given by

    .. math::

        \vec{v} = -\frac{ ∇ p × \vec{B} }{ q n |B|^2 }

    This is the velocity component of a fluid element perpendicular to
    the magnetic field.

    Examples
    --------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> dp = np.array([0, -1, 0]) * u.Pa / u.m
    >>> b = np.array([0, 0, 1]) * u.T
    >>> n = 1 / u.m**3
    >>> q = 1 * u.C
    >>> diamagnetic_drift(dp, b, n, q)
    <Quantity [ 1., -0., -0.] m / s>
    """
    cross = np.cross(dp, B)
    return -cross / q / n / (B * B).sum(-1)


vd_ = diamagnetic_drift
"""Alias to `~plasmapy.formulary.drifts.diamagnetic_drift`."""


@validate_quantities
def ExB_drift(E: u.Quantity[u.V / u.m], B: u.Quantity[u.T]) -> u.Quantity[u.m / u.s]:
    r"""
    Calculate the "electric cross magnetic" particle drift.

    **Aliases:** `veb_`

    Parameters
    ----------
    E : `~astropy.units.Quantity`
        Electric field vector
    B : `~astropy.units.Quantity`
        Magnetic field vector

    Returns
    -------
    v: `~astropy.units.Quantity`
        Drift velocity, in m/s

    Notes
    -----
    The E × B drift is given by

    .. math::

        \vec{v} = \frac{\vec{E} × \vec{B}}{|B|^2}

    and is independent of particle charge.

    Examples
    --------
    >>> import astropy.units as u
    >>> ex = np.array([1, 0, 0])
    >>> ey = np.array([0, 1, 0])
    >>> ExB_drift(ex * u.V / u.m, ey * u.T)
    <Quantity [0., 0., 1.] m / s>
    >>> ExB_drift(ex * u.V / u.m, ex * u.T)
    <Quantity [0., 0., 0.] m / s>
    >>> ExB_drift(ex * u.V / u.m, 100 * ey * u.T)
    <Quantity [0.  , 0.  , 0.01] m / s>
    """

    # np.cross drops units right now, thus this hack: see
    # https://github.com/PlasmaPy/PlasmaPy/issues/59
    cross = np.cross(E.si.value, B.si.value) * E.unit * B.unit
    return cross / (B * B).sum(-1)


veb_ = ExB_drift
"""Alias to `~plasmapy.formulary.drifts.ExB_drift`."""


@validate_quantities
def force_drift(
    F: u.Quantity[u.N], B: u.Quantity[u.T], q: u.Quantity[u.C]
) -> u.Quantity[u.m / u.s]:
    r"""
    Calculate the general force drift for a particle in a magnetic field.

    **Aliases:** `vfd_`

    Parameters
    ----------
    F : `~astropy.units.Quantity`
        Force acting on particle
    B : `~astropy.units.Quantity`
        Magnetic field
    q : `~astropy.units.Quantity`
        Particle charge

    Returns
    -------
    v: `~astropy.units.Quantity`
        Drift velocity, in m/s

    Notes
    -----
    The particle drift in a magnetic field and with a general
    force (e.g. gravity) applied to it is given by

    .. math::

        \vec{v} = \frac{\vec{F} × \vec{B}}{q |B|^2}

    Note the charge dependency.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants.si import g0, e, m_e
    >>> ex = np.array([1, 0, 0])
    >>> ez = np.array([0, 0, 1])
    >>> force_drift(-ez * g0 * m_e, ex * 0.01 * u.T, e)
    <Quantity [ 0.0000000e+00, -5.5756984e-09,  0.0000000e+00] m / s>
    >>> force_drift(-ez * g0 * m_e, ez * 0.01 * u.T, e)
    <Quantity [ 0., -0.,  0.] m / s>
    >>> force_drift(-ez * g0 * m_e, ex * u.T, e)
    <Quantity [ 0.0000000e+00, -5.5756984e-11,  0.0000000e+00] m / s>
    """
    cross = np.cross(F.si.value, B.si.value) * F.unit * B.unit
    return cross / (q * (B * B).sum(-1))


vfd_ = force_drift
"""Alias to `~plasmapy.formulary.drifts.force_drift`."""
