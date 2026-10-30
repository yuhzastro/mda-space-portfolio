from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
from linkbudget.link import (fspl_db, parabolic_gain_dbi, hpbw_deg, K_B_DB, LinkBudget,
                             system_noise_temp, pointing_loss_db)
from linkbudget.dvbs2 import MODCODS, select_modcod


def test_boltzmann():
    assert K_B_DB == pytest.approx(-228.60, abs=0.005)


def test_fspl_textbook_geo_ku():
    # GEO, 12 GHz, 36 000 km: L = 20 log10(4 pi d f / c) = 205.15 dB
    assert fspl_db(36_000e3, 12e9) == pytest.approx(205.15, abs=0.01)
    # inverse-square law: +6.02 dB per doubling of range
    assert fspl_db(2e6, 20e9) - fspl_db(1e6, 20e9) == pytest.approx(20 * np.log10(2))


def test_dish_gain_and_beamwidth():
    # 1 m, 20 GHz, 60 %: G = 10 log10(0.6 (pi D / lambda)^2) = 44.21 dBi
    assert parabolic_gain_dbi(1.0, 20e9, 0.6) == pytest.approx(44.21, abs=0.01)
    assert hpbw_deg(1.0, 20e9) == pytest.approx(70 * 0.0149896, rel=1e-5)
    # loss at the -3 dB edge of a 3 dB beamwidth is 3 dB
    assert pointing_loss_db(0.5, 1.0) == pytest.approx(3.0)


def test_noise_temperature_with_feed_loss():
    # 0 dB loss: Tsys = Tant + Tlna; infinite loss: Tsys -> T0 + Tlna
    assert system_noise_temp(50, 100, 0.0) == pytest.approx(150)
    assert system_noise_temp(50, 100, 60.0) == pytest.approx(390, rel=1e-5)


def test_link_budget_identity():
    lb = LinkBudget(f_hz=20e9, eirp_dbw=40, g_rx_dbi=40, t_sys_k=200, range_m=1e6,
                    symbol_rate=1e8, l_atm_db=1.0)
    expect = 40 - fspl_db(1e6, 20e9) - 1.0 + (40 - 10 * np.log10(200)) + 228.599
    assert lb.c_over_n0 == pytest.approx(expect, abs=1e-3)
    assert lb.es_over_n0 == pytest.approx(expect - 80, abs=1e-3)


def test_dvbs2_table_shape():
    assert len(MODCODS) == 28
    # Shannon: required Es/N0 must exceed capacity bound 2^eta - 1 (complex AWGN)
    for m in MODCODS:
        assert 10 ** (m.esn0_db / 10) > 2 ** m.eff - 1


def test_acm_selection():
    assert select_modcod(-10.0) is None
    m = select_modcod(10.0, margin_db=1.0, implementation_loss_db=0.5)
    assert m.esn0_db <= 8.5 and m.name == "8PSK 3/4"   # 16APSK 2/3 needs 8.97 dB


orbitkit = pytest.importorskip("orbitkit")


def test_doppler_sign_and_magnitude_over_pass():
    from orbitkit.tle import Satellite
    from orbitkit.passes import GroundStation, find_passes
    from linkbudget.doppler import doppler_profile
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    sat = Satellite.from_elements(t0, 1015.0, 98.98, raan_deg=200.0)
    gs = GroundStation("Montreal", 45.405, -73.95, 0.03, 10.0)
    p = max(find_passes(sat, gs, t0, 24.0), key=lambda x: x.max_el_deg)
    ts, el, rng, fd, fdd = doppler_profile(sat, gs, p.aos, p.los, 19.7e9)
    assert fd[0] > 0 > fd[-1]                       # approaching then receding
    # |f_D| <= f0 v / c with v ~ 7.35 km/s at 1015 km
    assert np.max(np.abs(fd)) < 19.7e9 * 7.35 / 299792.458
    # max |Doppler rate| at closest approach
    k = int(np.argmax(np.abs(fdd)))
    assert abs(ts[k] - (p.tca - p.aos).total_seconds()) < 30
