import numpy as np
import pytest
from scipy.linalg import expm
from spacearm.proximity import (cw_stm, cw_matrix, two_impulse_transfer, drift_free_vy,
                                propagate_nonlinear, inertial_to_hill, hill_to_inertial)
from spacearm.constants import MU_EARTH, R_EARTH

A_ISS = R_EARTH + 420e3
N = np.sqrt(MU_EARTH / A_ISS ** 3)
RC0 = np.array([A_ISS, 0, 0.0])
VC0 = np.array([0, np.sqrt(MU_EARTH / A_ISS) * np.cos(np.radians(51.6)),
                np.sqrt(MU_EARTH / A_ISS) * np.sin(np.radians(51.6))])


def test_stm_equals_matrix_exponential():
    for t in (10.0, 1000.0, 5000.0):
        assert np.allclose(cw_stm(N, t), expm(cw_matrix(N) * t), atol=1e-9)


def test_stm_group_property():
    assert np.allclose(cw_stm(N, 700) @ cw_stm(N, 300), cw_stm(N, 1000), atol=1e-10)


def test_drift_free_relative_orbit_closes():
    x0 = np.array([100.0, 0, 0, 0, drift_free_vy(N, 100.0), 0])
    X = cw_stm(N, 2 * np.pi / N) @ x0
    assert np.allclose(X, x0, atol=1e-8)
    # 2:1 ellipse: along-track amplitude = 2 * radial amplitude
    ys = [(cw_stm(N, t) @ x0)[1] for t in np.linspace(0, 2 * np.pi / N, 400)]
    assert (max(ys) - min(ys)) / 2 == pytest.approx(200.0, rel=1e-3)


def test_radial_offset_drift_rate():
    """Deputy 1 km below with zero relative velocity drifts ahead by
    $|\\Delta y| = 12\\pi |x_0|$ per orbit, from $y(T) = 6(\\sin nT - nT)x_0$."""
    x0 = np.array([-1000.0, 0, 0, 0, 0, 0])
    T = 2 * np.pi / N
    y = (cw_stm(N, T) @ x0)[1]
    assert y == pytest.approx(6 * (0 - 2 * np.pi) * -1000.0)      # = +12 pi km


def test_hill_frame_round_trip():
    rho, rhod = np.array([12.0, -30.0, 4.0]), np.array([0.01, -0.02, 0.005])
    rd, vd = hill_to_inertial(RC0, VC0, rho, rhod)
    r2, v2 = inertial_to_hill(RC0, VC0, rd, vd)
    assert np.allclose(r2, rho, atol=1e-8) and np.allclose(v2, rhod, atol=1e-11)


def test_cw_agrees_with_nonlinear_two_body_at_short_range():
    """At 100 m separation, CW linearisation error over one orbit is below 1 m
    (error scales as $\\rho^2 / r$)."""
    x0 = np.array([50.0, -100.0, 20.0, 0.02, -0.1, 0.0])
    ts = np.linspace(0, 2 * np.pi / N, 50)
    truth = propagate_nonlinear(RC0, VC0, x0[:3], x0[3:], ts)
    cw = np.array([cw_stm(N, t) @ x0 for t in ts])
    assert np.max(np.linalg.norm(truth[:, :3] - cw[:, :3], axis=1)) < 1.0


def test_two_impulse_rendezvous_to_hold_point():
    """V-bar approach from 2 km behind to the 250 m hold point (ISS 'approach
    ellipsoid' boundary is 2 x 1 x 1 km; 250 m is a typical V-bar hold)."""
    r0, v0 = np.array([0.0, -2000.0, 0.0]), np.zeros(3)
    rf, tf = np.array([0.0, -250.0, 0.0]), 0.5 * 2 * np.pi / N
    dv1, dv2 = two_impulse_transfer(N, r0, v0, rf, tf)
    X = cw_stm(N, tf) @ np.concatenate([r0, v0 + dv1])
    assert np.allclose(X[:3], rf, atol=1e-6)
    assert np.allclose(X[3:] + dv2, 0.0, atol=1e-9)
    # nonlinear truth: miss distance small relative to the 1.75 km transfer
    truth = propagate_nonlinear(RC0, VC0, r0, v0 + dv1, np.array([0.0, tf]))
    assert np.linalg.norm(truth[-1, :3] - rf) < 5.0
    # total delta-v: order of cm/s to tens of cm/s for a 1.75 km half-orbit hop
    assert 0.1 < np.linalg.norm(dv1) + np.linalg.norm(dv2) < 2.0


def test_radial_impulse_hop_matches_fehse():
    """Fehse, *Automated Rendezvous and Docking of Spacecraft* (2003), sec. 3.3:
    a half-orbit transfer along V-bar by two radial impulses needs
    $\\Delta v = n \\Delta y / 4$ for each impulse."""
    dy = 1750.0
    dv1, dv2 = two_impulse_transfer(N, np.array([0, -2000.0, 0]), np.zeros(3),
                                    np.array([0, -250.0, 0]), np.pi / N)
    assert abs(dv1[0]) == pytest.approx(N * dy / 4, rel=1e-9)
    assert abs(dv2[0]) == pytest.approx(N * dy / 4, rel=1e-9)
    assert np.allclose(dv1[1:], 0, atol=1e-12) and np.allclose(dv2[1:], 0, atol=1e-12)
