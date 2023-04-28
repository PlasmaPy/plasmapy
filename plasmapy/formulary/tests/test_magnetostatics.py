import numpy as np
import pytest

from astropy import constants
from astropy import units as u

from plasmapy.formulary.magnetostatics import (
    CircularWire,
    FiniteStraightWire,
    GeneralWire,
    InfiniteStraightWire,
    MagneticDipole,
)

mu0_4pi = constants.mu0 / 4 / np.pi


class Test_MagneticDipole:
    def setup_method(self):
        self.moment = np.array([0, 0, 1]) * u.A * u.m * u.m
        self.p0 = np.array([0, 0, 0]) * u.m

    def test_value_position1(self):
        "Test a known solution"
        p = np.array([1, 0, 0])
        B1 = MagneticDipole(self.moment, self.p0).magnetic_field(p=p)
        B1_expected = np.array([0, 0, -1]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_value_position2(self):
        "Test a known solution"
        p = np.array([0, 0, 1])
        B2 = MagneticDipole(self.moment, self.p0).magnetic_field(p=p)
        B2_expected = np.array([0, 0, 2]) * 1e-7 * u.T
        assert np.all(np.isclose(B2.value, B2_expected.value))
        assert B2.unit == u.T

    def test_value_cartesian1(self):
        "Test a known solution"
        B1 = MagneticDipole(self.moment, self.p0).magnetic_field(x=1, y=0, z=0)
        B1_expected = np.array([0, 0, -1]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_value_cartesian2(self):
        "Test a known solution"
        B2 = MagneticDipole(self.moment, self.p0).magnetic_field(x=0, y=0, z=1)
        B2_expected = np.array([0, 0, 2]) * 1e-7 * u.T
        assert np.all(np.isclose(B2.value, B2_expected.value))
        assert B2.unit == u.T

    def test_value_cylindrical1(self):
        "Test a known solution"
        B1 = MagneticDipole(self.moment, self.p0).magnetic_field(r=1, z=0, phi=0)
        B1_expected = np.array([0, 0, -1]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_value_cylindrical2(self):
        "Test a known solution"
        B2 = MagneticDipole(self.moment, self.p0).magnetic_field(r=0, z=1, phi=0)
        B2_expected = np.array([0, 0, 2]) * 1e-7 * u.T
        assert np.all(np.isclose(B2.value, B2_expected.value))
        assert B2.unit == u.T

    def test_value_spherical1(self):
        "Test a known solution"
        B1 = MagneticDipole(self.moment, self.p0).magnetic_field(
            r=1, theta=np.pi / 2.0, phi=0
        )
        B1_expected = np.array([0, 0, -1]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_value_spherical2(self):
        "Test a known solution"
        B2 = MagneticDipole(self.moment, self.p0).magnetic_field(r=1, theta=0, phi=0)
        B2_expected = np.array([0, 0, 2]) * 1e-7 * u.T
        assert np.all(np.isclose(B2.value, B2_expected.value))
        assert B2.unit == u.T

    def test_repr(self):
        "Test __repr__ function"
        B1 = MagneticDipole(self.moment, self.p0)
        assert repr(B1) == r"MagneticDipole(moment=[0. 0. 1.]A m2, p0=[0. 0. 0.]m)"


class Test_GeneralWire:
    def setup_method(self):
        self.cw = CircularWire(
            np.array([0, 0, 1]), np.array([0, 0, 0]) * u.m, 1 * u.m, 1 * u.A
        )
        p1 = np.array([0.0, 0.0, 0.0]) * u.m
        p2 = np.array([0.0, 0.0, 1.0]) * u.m
        self.fw = FiniteStraightWire(p1, p2, 1 * u.A)

    def test_not_callable(self):
        "Test that `GeneralWire` raises `TypeError` if its first argument is not callable"
        with pytest.raises(TypeError):
            GeneralWire("wire", 0, 1, 1 * u.A)

    def test_close_cw(self):
        "Test if the GeneralWire is close to the CircularWire it converted from"
        gw_cw = self.cw.to_GeneralWire()
        p = np.array([0, 0, 0])
        B_cw = self.cw.magnetic_field(p)
        B_gw_cw = gw_cw.magnetic_field(p)

        assert np.all(np.isclose(B_cw.value, B_gw_cw.value))
        assert B_cw.unit == B_gw_cw.unit

    def test_repr(self):
        "Test __repr__ function"
        gw_cw = self.cw.to_GeneralWire()
        # round numbers to avoid calculation accuracy mismatch
        gw_cw.t1 = -3.1516
        gw_cw.t2 = +3.1516
        assert (
            repr(gw_cw)
            == r"GeneralWire(parametric_eq=curve, t1=-3.1516, t2=3.1516, current=1.0A)"
        )

    def test_close_fw(self):
        "Test if the GeneralWire is close to the FiniteWire it converted from"
        gw_fw = self.fw.to_GeneralWire()
        p = np.array([1, 0, 0])
        B_fw = self.fw.magnetic_field(p)
        B_gw_fw = gw_fw.magnetic_field(p)

        assert np.all(np.isclose(B_fw.value, B_gw_fw.value))
        assert B_fw.unit == B_gw_fw.unit

    def test_value_error(self):
        "Test GeneralWire raise ValueError when argument t1>t2"
        with pytest.raises(ValueError):
            GeneralWire(lambda t: [0, 0, t], 2, 1, 1.0 * u.A)


class Test_FiniteStraightWire:
    def setup_method(self):
        self.p1 = np.array([0.0, 0.0, -1.0]) * u.m
        self.p2 = np.array([0.0, 0.0, 1.0]) * u.m
        self.current = 1 * u.A

    def test_same_point(self):
        "Test that `FiniteStraightWire` raises `ValueError` if p1 == p2"
        with pytest.raises(ValueError):
            FiniteStraightWire(self.p1, self.p1, self.current)

    def test_value1(self):
        "Test a known solution"
        fw = FiniteStraightWire(self.p1, self.p2, self.current)
        B1 = fw.magnetic_field([1, 0, 0])
        B1_expected = np.array([0, np.sqrt(2), 0]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_repr(self):
        "Test __repr__ function"
        fw = FiniteStraightWire(self.p1, self.p2, self.current)
        assert (
            repr(fw)
            == r"FiniteStraightWire(p1=[ 0.  0. -1.]m, p2=[0. 0. 1.]m, current=1.0A)"
        )


class Test_InfiniteStraightWire:
    def setup_method(self):
        self.direction = np.array([0, 1, 0])
        self.p0 = np.array([0, 0, 0]) * u.m
        self.current = 1 * u.A

    def test_value1(self):
        "Test a known solution"
        iw = InfiniteStraightWire(self.direction, self.p0, self.current)
        B1 = iw.magnetic_field([1, 0, 0])
        B1_expected = np.array([0, 0, -2]) * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_repr(self):
        "Test __repr__ function"
        iw = InfiniteStraightWire(self.direction, self.p0, self.current)
        assert (
            repr(iw)
            == r"InfiniteStraightWire(direction=[0. 1. 0.], p0=[0. 0. 0.]m, current=1.0A)"
        )


class Test_CircularWire:
    def setup_method(self):
        self.normalz = np.array([0, 0, 1])
        self.normalx = np.array([1, 0, 0])
        self.center = np.array([0, 0, 0]) * u.m
        self.radius = 1 * u.m
        self.current = 1 * u.A

    def test_negative_radius(self):
        "Test that `FiniteStraightWire` raises `ValueError` if radius < 0"
        with pytest.raises(ValueError):
            CircularWire(self.normalz, self.center, -1.0 * u.m, self.current)

    def test_value1(self):
        "Test a known solution"
        cw = CircularWire(self.normalz, self.center, self.radius, self.current)
        B1 = cw.magnetic_field([0, 0, 1])
        B1_expected = np.array([0, 0, 1]) * 2 * np.pi / 2**1.5 * 1e-7 * u.T
        assert np.all(np.isclose(B1.value, B1_expected.value))
        assert B1.unit == u.T

    def test_value2(self):
        "Test a known solution"
        cw = CircularWire(self.normalx, self.center, self.radius, self.current)
        B2 = cw.magnetic_field([1, 0, 0])
        B2_expected = np.array([1, 0, 0]) * 2 * np.pi / 2**1.5 * 1e-7 * u.T
        assert np.all(np.isclose(B2.value, B2_expected.value))
        assert B2.unit == u.T

    def test_repr(self):
        "Test __repr__ function"
        cw = CircularWire(self.normalz, self.center, self.radius, self.current)
        assert (
            repr(cw)
            == r"CircularWire(normal=[0. 0. 1.], center=[0. 0. 0.]m, radius=1.0m, current=1.0A)"
        )
