"""Free-space optical link: Gaussian-beam diffraction (PHYS 408).

A TEM$_{00}$ beam with waist $w_0$ has Rayleigh range $z_R = \\pi w_0^2/\\lambda$,
radius $w(z) = w_0\\sqrt{1 + (z/z_R)^2}$ and far-field half-angle divergence
$\\theta = \\lambda/(\\pi w_0)$. A circular receiver of radius $a$ centred in the
beam collects $1 - e^{-2a^2/w^2}$ of the power.

The beam is a solution of the paraxial Helmholtz equation; the angular-spectrum
propagator $H(f_x, f_y) = e^{ikz\\sqrt{1 - \\lambda^2(f_x^2 + f_y^2)}}$ propagates any
field exactly (Fourier optics). The tests use it to check $w(z)$.
"""
from __future__ import annotations
import numpy as np


def rayleigh_range(w0, lam):
    return np.pi * w0 ** 2 / lam


def beam_radius(z, w0, lam):
    return w0 * np.sqrt(1 + (z / rayleigh_range(w0, lam)) ** 2)


def collected_fraction(a, w):
    return 1 - np.exp(-2 * a ** 2 / w ** 2)


def angular_spectrum_propagate(U, dx, lam, z):
    n = U.shape[0]
    fx = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(fx, fx, indexing="ij")
    arg = 1 - (lam * FX) ** 2 - (lam * FY) ** 2
    H = np.exp(1j * 2 * np.pi / lam * z * np.sqrt(np.maximum(arg, 0))) * (arg > 0)
    return np.fft.ifft2(np.fft.fft2(U) * H)


def second_moment_radius(I, dx):
    """$w = 2\\sqrt{\\langle x^2\\rangle}$, the $1/e^2$ radius for a Gaussian."""
    n = I.shape[0]
    x = (np.arange(n) - n // 2) * dx
    X, _ = np.meshgrid(x, x, indexing="ij")
    return 2 * np.sqrt((I * X ** 2).sum() / I.sum())


def downlink_efficiency(range_m, lam, w0, rx_diameter, el_rad, zenith_transmittance=0.85,
                        rx_efficiency=0.5, pointing_loss=0.8):
    """Channel transmittance $\\eta$: diffraction capture, atmosphere
    (Beer-Lambert with airmass $1/\\sin\\varepsilon$), pointing, and receiver optics/detector."""
    w = beam_radius(range_m, w0, lam)
    eta_geo = collected_fraction(rx_diameter / 2, w)
    eta_atm = zenith_transmittance ** (1 / np.sin(el_rad))
    return eta_geo * eta_atm * pointing_loss * rx_efficiency
