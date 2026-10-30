import numpy as np
import pytest
from spaceenv.constants import Q_E, M_P, M_E, R_EARTH, C, B0_DIPOLE
from spaceenv.radiation import (dipole_B, B_mag, speed, gamma_from_kinetic, mirror_latitude, loss_cone,
                                bounce_period, drift_period, stormer_vertical_cutoff, boris_push)


def test_dipole_field_magnitude_and_divergence_free():
    for lat in np.radians([0, 20, 45, 70]):
        r = 3.0 * R_EARTH * np.array([np.cos(lat), 0, np.sin(lat)])
        assert np.linalg.norm(dipole_B(r)) == pytest.approx(B_mag(3.0, lat), rel=1e-12)
    # div B = 0 by central differences
    r0, h = np.array([2.1, 0.7, 1.3]) * R_EARTH, 1.0
    div = sum((dipole_B(r0 + h * e)[i] - dipole_B(r0 - h * e)[i]) / (2 * h) for i, e in enumerate(np.eye(3)))
    assert abs(div) < 1e-6 * np.linalg.norm(dipole_B(r0)) / R_EARTH


def test_relativistic_electron():
    W = 1e6 * Q_E                          # 1 MeV electron
    assert gamma_from_kinetic(W, M_E) == pytest.approx(2.9569, rel=1e-4)
    assert speed(W, M_E) / C == pytest.approx(0.94108, rel=1e-4)


def test_mirror_latitude_limits():
    assert mirror_latitude(np.pi / 2) == 0.0
    # sin^2(a) = cos^6(l)/sqrt(1+3 sin^2 l): check the root
    a = np.radians(30)
    l = mirror_latitude(a)
    assert np.cos(l) ** 6 / np.sqrt(1 + 3 * np.sin(l) ** 2) == pytest.approx(np.sin(a) ** 2, rel=1e-10)


def test_loss_cone_shrinks_with_L():
    # dipole: sin^2 a_lc = B_eq / B_foot, about 8.6 deg at L = 3 and 2.5 deg at L = 6.6
    assert loss_cone(3.0) > loss_cone(4.0) > loss_cone(6.6)
    assert 2.0 < np.degrees(loss_cone(6.6)) < 6.0


def test_stormer_cutoff_equator_surface():
    # ~ 14-15 GV at the magnetic equator at the surface (Stormer, with this B0: c B0 R_E / 4)
    assert stormer_vertical_cutoff(1.0, 0.0) / 1e9 == pytest.approx(14.23, abs=0.05)
    assert stormer_vertical_cutoff(1.0, np.radians(60)) < 1e9                       # polar cap: low cutoff


@pytest.fixture(scope="module")
def proton_orbit():
    """Full-orbit (Lorentz force) integration of a 10 MeV proton at L = 3, alpha_eq = 45 deg."""
    L, W, a = 3.0, 10e6 * Q_E, np.radians(45)
    v = speed(W, M_P)
    x0 = np.array([L * R_EARTH, 0.0, 0.0])
    v0 = v * np.array([0.0, np.sin(a), np.cos(a)])
    ts, xs, vs = boris_push(x0, v0, Q_E, M_P, 40.0, 5e-4, record_every=5)
    return L, W, a, v, ts, xs, vs


def test_boris_conserves_energy(proton_orbit):
    *_, v, ts, xs, vs = proton_orbit
    assert np.ptp(np.linalg.norm(vs, axis=1)) / v < 1e-10


def test_bounce_period_matches_guiding_centre_theory(proton_orbit):
    L, W, a, v, ts, xs, vs = proton_orbit
    cr = np.where(np.diff(np.sign(xs[:, 2])) != 0)[0]
    tb = np.mean(np.diff(ts[cr][::2]))
    # finite gyroradius (r_g / L R_E ~ 2 %) limits agreement to a few percent
    assert tb == pytest.approx(bounce_period(L, W, M_P, a), rel=0.04)


def test_drift_period_and_direction(proton_orbit):
    L, W, a, v, ts, xs, vs = proton_orbit
    phi = np.unwrap(np.arctan2(xs[:, 1], xs[:, 0]))
    rate = np.polyfit(ts, phi, 1)[0]
    assert rate < 0                                        # protons drift westward
    assert 2 * np.pi / abs(rate) == pytest.approx(drift_period(L, W, M_P, a), rel=0.04)


def test_mirror_latitude_from_full_orbit(proton_orbit):
    L, W, a, v, ts, xs, vs = proton_orbit
    lat = np.arcsin(xs[:, 2] / np.linalg.norm(xs, axis=1))
    assert np.degrees(lat.max()) == pytest.approx(np.degrees(mirror_latitude(a)), abs=1.5)
