from datetime import datetime, timedelta, timezone
import numpy as np
from orbitkit.tle import Satellite
from orbitkit.passes import GroundStation, find_passes
from orbitkit.coverage import WalkerShell, coverage_fraction

ISS = ("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")
GS = GroundStation("MDA Sainte-Anne-de-Bellevue", 45.405, -73.95, 0.03, 10.0)


def test_iss_passes_are_physical():
    sat = Satellite(*ISS)
    ps = find_passes(sat, GS, sat.epoch, 24.0)
    assert len(ps) >= 3
    for p in ps:
        assert p.aos < p.tca < p.los
        assert 10.0 < p.max_el_deg <= 90.0
        # LEO at ~350 km above a 10 deg mask: passes last < 9 min
        assert 0 < p.duration_s < 9 * 60
        # elevation equals mask at AOS/LOS
        for t in (p.aos, p.los):
            assert abs(np.degrees(GS.look(sat, t)[1]) - 10.0) < 1e-3


def test_range_rate_matches_finite_difference():
    sat = Satellite(*ISS)
    p = find_passes(sat, GS, sat.epoch, 24.0)[0]
    t, dt = p.aos + timedelta(seconds=60), 0.5
    rdot = GS.look(sat, t)[3]
    fd = (GS.look(sat, t + timedelta(seconds=dt))[2] - GS.look(sat, t - timedelta(seconds=dt))[2]) / (2 * dt)
    assert abs(rdot - fd) < 1e-4    # 0.1 m/s


def test_walker_geometry():
    sh = WalkerShell(1015.0, 98.98, 78, 6, 1)
    els = sh.elements()
    assert len(els) == 78
    r = sh.positions_ecef(0.0)
    assert np.allclose(np.linalg.norm(r, axis=1), 6378.137 + 1015.0)


def test_polar_shell_covers_poles_continuously():
    sh = WalkerShell(1015.0, 98.98, 78, 6, 1)
    cov = coverage_fraction([sh], [85.0, -85.0], [0.0, 90.0], np.arange(0, 6000, 120), 10.0)
    assert np.all(cov == 1.0)


def test_from_elements_altitude_and_period():
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    sat = Satellite.from_elements(t0, 1015.0, 98.98)
    assert sat.epoch == t0
    r0 = np.linalg.norm(sat.teme(t0)[0])
    assert abs(r0 - (6378.135 + 1015.0)) < 15.0      # mean vs osculating radius
    # Keplerian period for a = 7393 km is 105.4 min; J2 changes it by < 0.1 %
    T = 2 * np.pi * np.sqrt((6378.135 + 1015.0) ** 3 / 398600.8)
    u = lambda t: sat.teme(t)[0]
    r1 = u(t0 + timedelta(seconds=T))
    assert np.linalg.norm(r1 - u(t0)) < 0.002 * 2 * np.pi * 7393
