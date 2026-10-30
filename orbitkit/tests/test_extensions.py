"""Tests for the course-driven extensions: potential theory, relativity,
solar geometry and eclipses, and batch orbit determination."""
from datetime import datetime
import numpy as np
import pytest
from orbitkit.j2 import accel_j2, potential_j2
from orbitkit.relativity import (fractional_rate_offset, split_offsets, factory_frequency,
                                 eccentricity_correction, MU_GPS, C)
from orbitkit.sun import (sun_position_eci, sun_declination, shadow_fraction, eclipse_fraction_cylindrical,
                          beta_angle, AU_KM, R_EARTH_KM)
from orbitkit.kepler import coe2rv
from orbitkit.od import Station, measure, batch_least_squares
from orbitkit.j2 import propagate_cowell

A_GPS = 26_561.75e3      # m, GPS nominal semi-major axis


# --- MATH 217: acceleration is minus the gradient of the potential ---------------

@pytest.mark.parametrize("r", [[7000.0, 0, 0], [3000.0, -4000.0, 5000.0], [100.0, 200.0, 6900.0]])
def test_j2_acceleration_is_negative_gradient_of_potential(r):
    r = np.array(r)
    h = 1e-2                                   # km; larger steps lose to truncation, smaller to round-off
    grad = np.array([(potential_j2(r + h * e) - potential_j2(r - h * e)) / (2 * h) for e in np.eye(3)])
    assert np.allclose(accel_j2(r), -grad, rtol=1e-8, atol=1e-12)


def test_potential_curl_free_numerically():
    """$\\nabla\\times a = 0$ for a conservative field (mixed partials commute)."""
    r0, h = np.array([4000.0, 3000.0, 5000.0]), 1e-2
    J = np.array([(accel_j2(r0 + h * e) - accel_j2(r0 - h * e)) / (2 * h) for e in np.eye(3)]).T
    assert np.allclose(J, J.T, atol=1e-13)


# --- PHYS 200: GPS relativistic clock offsets ------------------------------------

def test_gps_clock_offsets_published_values():
    grav, vel = split_offsets(A_GPS)
    assert grav * 86400e6 == pytest.approx(45.7, abs=0.1)        # us/day, gravitational
    assert vel * 86400e6 == pytest.approx(-7.2, abs=0.05)        # us/day, special relativity
    assert fractional_rate_offset(A_GPS) * 86400e6 == pytest.approx(38.6, abs=0.1)
    # IS-GPS-200 sec. 3.3.1.1: 10.23 MHz clocks are set to 10.22999999543 MHz
    assert factory_frequency(10.23e6, A_GPS) == pytest.approx(10.22999999543e6, abs=1e-5)


def test_gps_eccentricity_constant_F():
    # IS-GPS-200 sec. 20.3.3.3.3.1: F = -4.442807633e-10 s / sqrt(m)
    F = eccentricity_correction(1.0, 1.0, np.pi / 2) / 1.0
    assert F == pytest.approx(-4.442807633e-10, rel=1e-9)
    # e = 0.02 gives up to ~45 ns (~14 m of range): must be applied by receivers
    assert abs(eccentricity_correction(A_GPS, 0.02, np.pi / 2)) == pytest.approx(45.8e-9, rel=0.01)


# --- ASTR 205: solar position and eclipses ----------------------------------------

def test_sun_vallado_example_5_1():
    # Vallado, Fundamentals 4th ed., Ex. 5-1: 2006-04-02 00:00 UT1
    s = sun_position_eci(datetime(2006, 4, 2)) / AU_KM
    assert np.allclose(s, [0.9771945, 0.1924424, 0.0834308], atol=5e-6)


@pytest.mark.parametrize("t,dec", [(datetime(2024, 3, 20, 3, 6), 0.0),
                                   (datetime(2024, 6, 20, 20, 51), 23.436),
                                   (datetime(2024, 12, 21, 9, 20), -23.436)])
def test_equinox_and_solstice_declination(t, dec):
    assert np.degrees(sun_declination(t)) == pytest.approx(dec, abs=0.01)


def test_shadow_function_regions():
    s = np.array([AU_KM, 0.0, 0.0])
    assert shadow_fraction([7000.0, 0, 0], s) == 1.0                     # day side
    assert shadow_fraction([-7000.0, 0, 0], s) == 0.0                    # umbra
    # penumbra crossing is monotonic between umbra and sunlight
    ys = np.linspace(R_EARTH_KM - 100, R_EARTH_KM + 100, 400)
    nu = [shadow_fraction([-7000.0, y, 0.0], s) for y in ys]
    assert np.all(np.diff(nu) >= -1e-12) and nu[0] == 0.0 and nu[-1] == 1.0


def test_eclipse_duration_matches_cylindrical_formula():
    """Integrate the conical shadow function along a beta = 0 orbit; the
    shadow fraction matches the cylindrical formula to within the penumbra width."""
    a = R_EARTH_KM + 1015.0
    s = np.array([AU_KM, 0.0, 0.0])
    nu = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    frac = 1 - np.mean([shadow_fraction(a * np.array([np.cos(u), np.sin(u), 0.0]), s) for u in nu])
    assert frac == pytest.approx(eclipse_fraction_cylindrical(a, 0.0), abs=2e-3)
    assert eclipse_fraction_cylindrical(a, 0.0) == pytest.approx(np.arcsin(R_EARTH_KM / a) / np.pi, rel=1e-9)
    # no eclipse above the critical beta angle
    assert eclipse_fraction_cylindrical(a, np.arcsin(R_EARTH_KM / a) + 1e-3) == 0.0


def test_beta_angle_sun_synchronous_dawn_dusk():
    s = np.array([AU_KM, 0.0, 0.0])
    r, v = coe2rv(7000.0, 0.0, np.radians(90.0), np.radians(90.0), 0.0, 0.0)   # orbit normal along +x
    assert np.degrees(abs(beta_angle(r, v, s))) == pytest.approx(90.0, abs=1e-9)


# --- PHYS 229: batch least squares orbit determination ----------------------------

@pytest.fixture(scope="module")
def od_case():
    rng = np.random.default_rng(2026)
    r0, v0 = coe2rv(6378.137 + 800.0, 0.001, np.radians(98.6), np.radians(30.0), 0.0, np.radians(10.0))
    times = np.arange(0.0, 6 * 3600.0, 20.0)
    _, R, V = propagate_cowell(r0, v0, (0, times[-1]), t_eval=times)
    stations = [Station(45.405, -73.95), Station(53.2, -105.1), Station(78.2, 15.4), Station(-33.9, 18.4)]
    obs = measure(np.hstack([R, V]), times, stations)
    s_rho, s_rd = 0.010, 1e-5                       # 10 m, 1 cm/s
    obs = [(k, j, rho + s_rho * rng.normal(), rd + s_rd * rng.normal()) for k, j, rho, rd in obs]
    guess = np.concatenate([r0 + [5.0, -3.0, 2.0], v0 + [0.003, -0.002, 0.001]])
    res = batch_least_squares(guess, times, obs, stations, s_rho, s_rd)
    return np.concatenate([r0, v0]), res


def test_od_converges(od_case):
    truth, res = od_case
    assert res.iterations < 10
    assert np.linalg.norm(res.x0[:3] - truth[:3]) < 0.05           # 50 m from a 6 km initial error


def test_od_residuals_are_chi_square_consistent(od_case):
    _, res = od_case
    # chi^2 / dof within 4 standard deviations of 1 (sd = sqrt(2/dof))
    assert abs(res.chi2 / res.dof - 1) < 4 * np.sqrt(2 / res.dof)


def test_od_covariance_is_realistic(od_case):
    """Normalised estimation error squared, $e^TP^{-1}e$, follows $\\chi^2_6$:
    below 22.46 with 99.9 % probability."""
    truth, res = od_case
    e = res.x0 - truth
    assert e @ np.linalg.solve(res.P, e) < 22.46
