"""End-to-end focusing tests of the Range-Doppler Algorithm."""
import numpy as np
import pytest
from sarsim.rda import SARParams, simulate_raw, rda, C
from sarsim.chirp import irf_metrics


@pytest.fixture(scope="module")
def focused():
    p = SARParams()
    # second target on a range sample and an azimuth sample, so the sampled
    # peak is the true peak (off-grid targets lose up to sinc(B/2fs) in amplitude)
    tg = [(p.R_ref, 0.0, 1.0), (p.range_axis()[p.n_rg // 2 + 360], 170 / p.prf, 0.5 * np.exp(1j))]
    return p, tg, rda(p, simulate_raw(p, tg))


def _peak(p, img, R0, eta_c):
    R, eta = p.range_axis(), p.az_axis()
    i, j = np.argmin(abs(eta - eta_c)), np.argmin(abs(R - R0))
    sub = np.abs(img[i - 20:i + 21, j - 20:j + 21])
    di, dj = np.unravel_index(sub.argmax(), sub.shape)
    return i, j, i - 20 + di, j - 20 + dj


def test_targets_focus_at_true_positions(focused):
    p, tg, img = focused
    for R0, ec, _ in tg:
        i, j, I, J = _peak(p, img, R0, ec)
        assert abs(I - i) <= 1 and abs(J - j) <= 1


def test_range_resolution_and_pslr(focused):
    p, tg, img = focused
    for R0, ec, _ in tg:
        _, _, I, J = _peak(p, img, R0, ec)
        w, pslr, _ = irf_metrics(img[I, J - 256:J + 256])
        assert w * C / (2 * p.fs) == pytest.approx(0.886 * C / (2 * p.B), rel=0.01)
        assert pslr == pytest.approx(-13.26, abs=0.2)


def test_azimuth_resolution_matches_antenna_limited_doppler_spectrum(focused):
    """Azimuth IRF predicted independently: the inverse FFT of the two-way
    antenna pattern mapped to Doppler, $f = 2V\\sin\\theta/\\lambda$. A defocused
    image (wrong phase history) would be much wider."""
    p, tg, img = focused
    f = np.fft.fftfreq(4096, 1 / p.prf)
    s = p.lam * f / (2 * p.Vr)
    W = np.sinc(p.La * s / p.lam) ** 2
    W[np.abs(s) > p.lam / p.La] = 0
    h = np.fft.fftshift(np.fft.ifft(W))
    w_pred = irf_metrics(h[2048 - 256:2048 + 256])[0]
    for R0, ec, _ in tg:
        _, _, I, J = _peak(p, img, R0, ec)
        w = irf_metrics(img[I - 256:I + 256, J])[0]
        assert w == pytest.approx(w_pred, rel=0.01)
        # same order as the classic L_a / 2 rule
        assert 0.6 * p.La / 2 < w * p.Vr / p.prf < 1.1 * p.La / 2


def test_relative_amplitude_and_phase_preserved(focused):
    p, tg, img = focused
    a = [img[_peak(p, img, R0, ec)[2], _peak(p, img, R0, ec)[3]] for R0, ec, _ in tg]
    # the image phase of a target is -4 pi f0 R0 / c + arg(amp)
    ph = [np.angle(x * np.exp(4j * np.pi * p.f0 * R0 / C)) for x, (R0, _, _) in zip(a, tg)]
    assert abs(a[1]) / abs(a[0]) == pytest.approx(0.5, rel=0.02)
    assert np.angle(np.exp(1j * (ph[1] - ph[0] - 1.0))) == pytest.approx(0.0, abs=0.1)


def test_image_snr_matches_matched_filter_theorem():
    """Matched-filter theorem: peak SNR = E / sigma^2. RDA's azimuth filter is
    phase-only (no amplitude weighting), so the measured SNR sits below that
    by the mismatch loss $(\\sum_f W)^2 / (N \\sum_f W^2)$ over the Doppler
    spectrum, including the aliased parts that end up in azimuth ambiguities."""
    p = SARParams(n_rg=1024, n_az=2048)
    sig = simulate_raw(p, [(p.R_ref, 0.0, 1.0)])
    noi = simulate_raw(p, [], noise_std=1.0, rng=np.random.default_rng(1))
    a, b = rda(p, sig), rda(p, noi)
    half = int(round(p.Tp * p.fs)) // 2                 # avoid partially-correlated range edges
    snr = np.abs(a).max() ** 2 / np.var(b[:, half + 20:p.n_rg - half - 20])
    E = np.sum(np.abs(sig) ** 2)
    f = np.fft.fftfreq(4096, 1 / p.prf)
    Ws = []
    for k in (-1, 0, 1):
        s = p.lam * (f + k * p.prf) / (2 * p.Vr)
        W = np.sinc(p.La * s / p.lam) ** 2
        W[np.abs(s) > p.lam / p.La] = 0
        Ws.append(W)
    loss = Ws[1].sum() ** 2 / (len(f) * sum((W ** 2).sum() for W in Ws))
    assert 10 * np.log10(snr / E) == pytest.approx(10 * np.log10(loss), abs=0.2)
