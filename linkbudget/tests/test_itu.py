"""ITU-R models. Coefficients are checked against values tabulated in
P.838-3, and the full P.618-13 procedure is cross-checked against the
independent `itur` implementation (del Portillo, MIT) where it is installed."""
import numpy as np
import pytest
from linkbudget.itu import rain_coefficients, rain_attenuation, sky_noise_from_attenuation


@pytest.mark.parametrize("f,kH,aH,kV,aV", [
    # P.838-3 tabulated coefficients (horizontal / vertical)
    (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
    (20.0, 0.09164, 1.0568, 0.09611, 0.9847),
    (30.0, 0.2403, 0.9485, 0.2291, 0.9129),
])
def test_p838_tabulated_values(f, kH, aH, kV, aV):
    _, _, (kh, ah, kv, av) = rain_coefficients(f)
    assert kh == pytest.approx(kH, rel=2e-3) and ah == pytest.approx(aH, abs=2e-4)
    assert kv == pytest.approx(kV, rel=2e-3) and av == pytest.approx(aV, abs=2e-4)


def test_p838_circular_is_mean_of_h_and_v():
    k, a, (kh, ah, kv, av) = rain_coefficients(20.0, 30.0, 45.0)
    assert k == pytest.approx((kh + kv) / 2)
    assert a == pytest.approx((kh * ah + kv * av) / (kh + kv))


def test_p618_monotonic():
    base = dict(lat_deg=45.4, h0_km=3.2, R001=33.0, hs_km=0.03)
    ps = [0.001, 0.01, 0.1, 1.0, 5.0]
    A = [rain_attenuation(20.0, 30.0, p=p, **base) for p in ps]
    assert all(x > y for x, y in zip(A, A[1:]))
    assert rain_attenuation(30.0, 30.0, **base) > rain_attenuation(20.0, 30.0, **base)
    assert rain_attenuation(20.0, 20.0, **base) > rain_attenuation(20.0, 60.0, **base)


def test_p618_no_rain_above_rain_height():
    assert rain_attenuation(20.0, 30.0, 45.4, 3.2, 33.0, hs_km=4.0) == 0.0


itur = pytest.importorskip("itur")


@pytest.mark.parametrize("f,el,p,R", [(20, 30, 0.01, 40), (20, 30, 0.1, 40), (30, 15, 1, 40),
                                      (12, 45, 0.5, 40), (20, 8, 0.01, 40), (28, 40, 0.01, 90)])
def test_p618_matches_itur_reference(f, el, p, R):
    from itur.models import itu618, itu839
    lat, lon = 45.405, -73.95
    h0 = float(itu839.isoterm_0(lat, lon).value)
    ours = rain_attenuation(f, el, lat, h0, R, p, hs_km=0.03, tau_deg=45)
    ref = itu618.rain_attenuation(lat, lon, f, el, hs=0.03, p=p, R001=R, tau=45).value
    assert ours == pytest.approx(float(ref), rel=1e-9)


def test_sky_noise_limits():
    assert sky_noise_from_attenuation(0.0) == pytest.approx(2.725)
    assert sky_noise_from_attenuation(100.0) == pytest.approx(275.0)
    assert sky_noise_from_attenuation(3.0) == pytest.approx(275 * (1 - 10 ** -0.3) + 2.725 * 10 ** -0.3)
