import numpy as np
import pytest
from scipy.special import jn_zeros
from sarsim.antenna import rect_pattern, circ_pattern, far_field_fft, hpbw, directivity_rect
from sarsim.chirp import chirp, matched_filter, irf_metrics, stationary_phase_spectrum
from sarsim.radar import received_power, point_target_snr, nesz, resolution, C, K_B

LAM = C / 5.405e9


def test_rect_hpbw_is_0886_lambda_over_L():
    # half-power point of sinc^2: x = 0.44295 -> full width 0.8859 lambda / L
    assert hpbw(rect_pattern, LAM, 15.0) / (LAM / 15.0) == pytest.approx(0.8859, abs=1e-4)


def test_airy_first_null_from_bessel_zero():
    D = 3.0
    j11 = jn_zeros(1, 1)[0]
    assert j11 == pytest.approx(3.8317, abs=1e-4)
    theta0 = np.arcsin(j11 / np.pi * LAM / D)          # 1.2197 lambda / D
    assert abs(circ_pattern(theta0, D, LAM)) < 1e-12
    assert np.arcsin(j11 / np.pi * LAM / D) / (LAM / D) == pytest.approx(1.2197, abs=1e-4)
    # Airy HPBW = 1.029 lambda / D
    assert hpbw(circ_pattern, LAM, D) / (LAM / D) == pytest.approx(1.029, abs=1e-3)


def test_far_field_is_fourier_transform_of_aperture():
    """Fraunhofer: FFT of a sampled uniform aperture reproduces the sinc pattern."""
    L, dx = 15.0, LAM / 8
    ap = np.ones(int(L / dx))
    s, E = far_field_fft(ap, dx, LAM)
    m = np.abs(s) < 3 * LAM / L
    assert np.allclose(E[m], np.abs(rect_pattern(np.arcsin(s[m]), L, LAM)), atol=2e-3)


def test_planar_aperture_gain_radarsat2_class():
    # 15 m x 1.5 m at 5.405 GHz, 100 % efficiency: 4 pi (22.5 m^2) / (0.05547 m)^2 = 49.63 dBi
    assert 10 * np.log10(directivity_rect(15, 1.5, LAM)) == pytest.approx(49.63, abs=0.01)
    # equals the product of the two principal-plane beamwidth factors, 4 pi / (theta_x theta_y) with theta = lambda/L
    assert directivity_rect(15, 1.5, LAM) == pytest.approx(4 * np.pi / ((LAM / 15) * (LAM / 1.5)))


def test_pulse_compression_width_pslr_gain():
    T, B, fs = 20e-6, 30e6, 120e6
    _, s = chirp(T, B, fs)
    y = matched_filter(np.concatenate([np.zeros(1000), s, np.zeros(1000)]), s)
    k = int(np.argmax(np.abs(y)))
    w, pslr, islr = irf_metrics(y[k - 512:k + 512])
    assert w / fs == pytest.approx(0.886 / B, rel=0.01)
    assert pslr == pytest.approx(-13.26, abs=0.15)
    assert islr == pytest.approx(-9.7, abs=0.4)
    # SNR gain of the matched filter for white noise is the number of pulse
    # samples, T fs (= TB when sampled at B)
    rng = np.random.default_rng(0)
    n = (rng.normal(size=8192) + 1j * rng.normal(size=8192)) / np.sqrt(2)
    yn = matched_filter(n, s)[len(s):-len(s)]
    gain = np.abs(y[k]) ** 2 / np.var(yn)
    assert 10 * np.log10(gain) == pytest.approx(10 * np.log10(T * fs), abs=0.2)


def test_stationary_phase_spectrum():
    """Exact FFT spectrum of a TB = 600 chirp vs. the stationary-phase approximation."""
    T, B, fs = 20e-6, 30e6, 120e6
    t, s = chirp(T, B, fs)
    n = 1 << 16
    S = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.pad(s, ((n - len(s)) // 2, (n - len(s)) // 2))))) / fs
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    m = np.abs(f) < 0.8 * B / 2                      # away from the Fresnel ripple at the band edges
    Ssp = stationary_phase_spectrum(f[m], T, B)
    assert np.median(np.abs(S[m]) / np.abs(Ssp)) == pytest.approx(1.0, abs=0.02)
    dphi = np.angle(S[m] * np.conj(Ssp))
    assert np.sqrt(np.mean(dphi ** 2)) < 0.1


def test_radar_equation_r4_and_snr_consistency():
    assert received_power(1, 100, LAM, 1, 2e5) / received_power(1, 100, LAM, 1, 1e5) == pytest.approx(1 / 16)
    # NESZ is the sigma0 that gives SNR = 1 for a resolution cell
    Pav, G, R, V, B, inc, T = 300.0, 10 ** 4.5, 900e3, 7100.0, 30e6, np.radians(35), 600.0
    F, L = 10 ** 0.3, 10 ** 0.5                     # 3 dB noise figure, 5 dB system losses
    s0 = nesz(Pav, G, LAM, R, V, B, inc, T, F, L)
    dgr, da = resolution(B, 15.0, inc)
    sigma = s0 * (15.0 / 2) * (C / (2 * B * np.sin(inc)))
    assert point_target_snr(Pav, G, LAM, sigma, R, V, 15.0, T, F, L) == pytest.approx(1.0, rel=1e-12)
    # RADARSAT-2-class numbers give an NESZ in the published -20 to -30 dB range
    assert -30 < 10 * np.log10(s0) < -18
