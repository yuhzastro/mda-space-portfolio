import numpy as np
import pytest
from linkbudget.noise import Stage, passive_loss, cascade_te, cascade_nf_db, T0
from linkbudget.optical import (rayleigh_range, beam_radius, collected_fraction, angular_spectrum_propagate,
                                second_moment_radius)
from linkbudget.qkd import bb84, h2, decoy_key_rate


# --- Friis ----------------------------------------------------------------------

def test_te_and_nf_cascades_agree():
    st = [Stage("LNA", 30, nf_db=0.8), passive_loss("cable", 2.0), Stage("downconverter", 10, nf_db=8)]
    assert cascade_nf_db(st) == pytest.approx(10 * np.log10(1 + cascade_te(st) / T0), abs=1e-12)


def test_passive_loss_noise_figure_equals_loss():
    assert cascade_nf_db([passive_loss("wg", 3.0)]) == pytest.approx(3.0, abs=1e-12)


def test_lna_first_beats_loss_first():
    """Why the LNA sits at the feed: 1 dB of loss before a 0.8 dB, 30 dB LNA
    raises T_e from 59 K to about 149 K; after it, by under 0.1 K."""
    lna, wg = Stage("LNA", 30, nf_db=0.8), passive_loss("wg", 1.0)
    assert cascade_te([lna]) == pytest.approx(T0 * (10 ** 0.08 - 1))
    assert cascade_te([wg, lna]) == pytest.approx(75.09 + 1.2589 * 58.66, abs=0.05)   # 148.9 K
    assert cascade_te([lna, wg]) - cascade_te([lna]) < 0.1


# --- Gaussian beams -------------------------------------------------------------

def test_gaussian_beam_formulae():
    lam, w0 = 785e-9, 0.05
    zR = rayleigh_range(w0, lam)
    assert beam_radius(zR, w0, lam) == pytest.approx(np.sqrt(2) * w0)
    assert beam_radius(1e3 * zR, w0, lam) / (1e3 * zR) == pytest.approx(lam / (np.pi * w0), rel=1e-6)
    assert collected_fraction(w0, w0) == pytest.approx(1 - np.exp(-2))     # 86.5 % inside 1/e^2


def test_angular_spectrum_reproduces_gaussian_spreading():
    """Fourier-optics propagation of a sampled Gaussian through 2 z_R matches $w(z)$."""
    lam, w0, n, dx = 1e-6, 1e-3, 512, 2.5e-5
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    U0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
    z = 2 * rayleigh_range(w0, lam)
    Uz = angular_spectrum_propagate(U0, dx, lam, z)
    assert second_moment_radius(np.abs(Uz) ** 2, dx) == pytest.approx(beam_radius(z, w0, lam), rel=1e-3)
    # power conserved (unitary propagation)
    assert np.sum(np.abs(Uz) ** 2) == pytest.approx(np.sum(np.abs(U0) ** 2), rel=1e-9)


# --- BB84 -----------------------------------------------------------------------

def test_bb84_no_eavesdropper_no_errors():
    a, b = bb84(4000, np.random.default_rng(0))
    assert np.all(a == b)
    assert len(a) / 4000 == pytest.approx(0.5, abs=0.03)        # sifting keeps half


def test_bb84_intercept_resend_gives_25_percent_qber():
    a, b = bb84(20000, np.random.default_rng(1), eve=True)
    assert np.mean(a != b) == pytest.approx(0.25, abs=0.015)


def test_bb84_misalignment_qber_is_sin_squared():
    th = np.radians(10)
    a, b = bb84(20000, np.random.default_rng(2), misalign=th)
    assert np.mean(a != b) == pytest.approx(np.sin(th) ** 2, abs=0.01)


def test_shor_preskill_threshold():
    """One-way BB84 key vanishes when $1 - 2h(e) = 0$: e = 11.0 %."""
    from scipy.optimize import brentq
    assert brentq(lambda e: 1 - 2 * h2(e), 0.01, 0.2) == pytest.approx(0.110, abs=5e-4)


def test_decoy_key_rate_behaviour():
    etas = 10 ** (-np.arange(0, 60, 1.0) / 10)
    R = decoy_key_rate(etas)
    nz = R > 0
    # R decreases with loss and drops to zero at a finite cutoff (dark counts)
    assert np.all(np.diff(R[nz]) < 0) and not nz[-1] and nz[0]
    # in the high-loss regime R is linear in eta: about 1 bit per 10 dB of extra loss per decade
    r = R[20] / R[30]
    assert 8 < r < 13
