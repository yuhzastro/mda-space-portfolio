"""Validation against published reference values.

[V06]  Vallado, Crawford, Hujsak, Kelso, 'Revisiting Spacetrack Report #3',
       AIAA 2006-6753, SGP4-VER.TLE / tcppver.out, and Appendix C example.
[V13]  Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
"""
from datetime import datetime, timedelta
import numpy as np
import pytest
from orbitkit.timeutil import jd_from_datetime, gmst_iau82
from orbitkit.frames import teme_to_ecef, geodetic_to_ecef, ecef_to_geodetic, look_angles
from orbitkit.tle import Satellite
from orbitkit.kepler import coe2rv, rv2coe
from orbitkit.j2 import secular_rates, propagate_cowell, sun_synchronous_inclination
from orbitkit.coverage import earth_central_angle

L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


def test_julian_date_j2000():
    assert jd_from_datetime(datetime(2000, 1, 1, 12)) == pytest.approx(2451545.0, abs=1e-9)


def test_gmst_vallado_example_3_5():
    # [V13] Ex. 3-5: 1992 Aug 20 12:14 UT1 -> GMST = 152.578787810 deg
    g = np.degrees(gmst_iau82(jd_from_datetime(datetime(1992, 8, 20, 12, 14))))
    assert g == pytest.approx(152.578787810, abs=1e-8)


@pytest.mark.parametrize("tsince,r_ref,v_ref", [
    (0.0, [7022.46529266, -1400.08296755, 0.03995155],
          [1.893841015, 6.405893759, 4.534807250]),
    (360.0, [-7154.03120202, -3783.17682504, -3536.19412294],
            [4.741887409, -4.151817765, -2.093935425]),
])
def test_sgp4_verification_vector_00005(tsince, r_ref, v_ref):
    # [V06] tcppver.out, satellite 00005
    sat = Satellite(L1, L2)
    _, r, v = sat.sat.sgp4_tsince(tsince)
    assert np.allclose(r, r_ref, atol=1e-6)      # 1 mm
    assert np.allclose(v, v_ref, atol=1e-9)


def test_teme_to_pef_vallado_2006_appendix():
    # [V06] App. C: 2004-04-06 07:51:28.386009 UTC, dUT1 = -0.4399619 s
    t_ut1 = datetime(2004, 4, 6, 7, 51, 28, 386009) - timedelta(seconds=0.4399619)
    r, v = teme_to_ecef([5094.18016210, 6127.64465950, 6380.34453270],
                        [-4.746131487, 0.785818041, 5.531931288],
                        gmst_iau82(jd_from_datetime(t_ut1)))
    assert np.allclose(r, [-1033.47503130, 7901.30558560, 6380.34453270], atol=1e-6)
    assert np.allclose(v, [-3.225632747, -2.872442511, 5.531931288], atol=1e-8)


def test_geodetic_round_trip():
    for lat, lon, h in [(45.405, -73.95, 0.03), (-89.9, 10, 500), (0, 179.9, 35786)]:
        r = geodetic_to_ecef(np.radians(lat), np.radians(lon), h)
        la, lo, hh = ecef_to_geodetic(r)
        assert np.degrees(la) == pytest.approx(lat, abs=1e-10)
        assert np.degrees(lo) == pytest.approx(lon, abs=1e-10)
        assert hh == pytest.approx(h, abs=1e-8)


def test_zenith_look_angle():
    lat, lon = np.radians(45.0), np.radians(-74.0)
    up = geodetic_to_ecef(lat, lon, 500.0)
    _, el, rng, _ = look_angles(up, np.zeros(3), lat, lon, 0.0)
    assert np.degrees(el) == pytest.approx(90.0, abs=1e-9)
    assert rng == pytest.approx(500.0, abs=1e-9)


def test_coe_rv_round_trip():
    # [V13] Ex. 2-5/2-6 elements: p=11067.79 km, e=0.83285, i=87.87 deg ...
    el = (11067.79 / (1 - 0.83285 ** 2), 0.83285, np.radians(87.87),
          np.radians(227.89), np.radians(53.38), np.radians(92.335))
    r, v = coe2rv(*el)
    assert np.allclose(r, [6525.368, 6861.532, 6449.119], atol=2e-3)   # [V13] Ex. 2-6
    assert np.allclose(v, [4.902279, 5.533140, -1.975710], atol=2e-6)
    assert np.allclose(rv2coe(r, v), el, rtol=1e-10)


def test_sun_synchronous_inclination_800km():
    # Standard result: circular SSO at 800 km has i = 98.6 deg
    assert np.degrees(sun_synchronous_inclination(6378.137 + 800)) == pytest.approx(98.60, abs=0.05)


def test_cowell_j2_raan_drift_matches_secular_theory():
    """Numerical J2 integration over 5 days: the mean RAAN drift (linear fit
    to osculating RAAN) must match first-order secular theory to < 0.5 %."""
    a, e, i = 6378.137 + 420.0, 0.0005, np.radians(51.64)
    r0, v0 = coe2rv(a, e, i, 0.3, 0.2, 0.0)
    T = 5 * 86400.0
    t, R, V = propagate_cowell(r0, v0, (0, T), t_eval=np.linspace(0, T, 4000))
    raan = np.unwrap([rv2coe(r, v)[3] for r, v in zip(R, V)])
    slope = np.polyfit(t, raan, 1)[0]
    # mean a differs from osculating a0 by O(J2); compare at 0.5 % level
    assert slope == pytest.approx(secular_rates(a, e, i)[0], rel=5e-3)
    # ISS nodal regression ~ -5.0 deg/day
    assert np.degrees(slope) * 86400 == pytest.approx(-5.0, abs=0.15)


def test_cowell_energy_conservation_two_body():
    r0, v0 = coe2rv(7000.0, 0.01, 1.0, 0.0, 0.0, 0.0)
    t, R, V = propagate_cowell(r0, v0, (0, 86400), t_eval=[0, 86400], j2=False)
    E = lambda r, v: 0.5 * v @ v - 398600.4418 / np.linalg.norm(r)
    assert abs(E(R[-1], V[-1]) - E(R[0], V[0])) < 1e-10


def test_earth_central_angle_geo():
    # GEO at 0 deg elevation: lambda = arccos(R/(R+h)) = 81.30 deg
    assert np.degrees(earth_central_angle(35786.0, 0.0)) == pytest.approx(81.30, abs=0.01)
