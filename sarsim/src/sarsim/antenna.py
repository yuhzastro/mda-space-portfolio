"""Aperture antenna far-field patterns (Fraunhofer diffraction).

In the far field the radiated field is the 2-D Fourier transform of the
aperture illumination (PHYS 408, Fourier optics). For a uniformly illuminated
aperture:

* rectangle of length $L$: $E(\\theta) \\propto \\mathrm{sinc}\\left(\\frac{L\\sin\\theta}{\\lambda}\\right)$,
  with $\\mathrm{sinc}(x) = \\sin(\\pi x)/(\\pi x)$, half-power beamwidth $0.886\\,\\lambda/L$,
  first null at $\\sin\\theta = \\lambda/L$;
* circle of diameter $D$: $E(\\theta) \\propto \\frac{2J_1(u)}{u}$, $u = \\frac{\\pi D}{\\lambda}\\sin\\theta$
  (Airy pattern). $J_1$ is the Bessel function of the first kind that solves
  $x^2y'' + xy' + (x^2-1)y = 0$ (MATH 316), and the first null is at
  $u = j_{1,1} = 3.8317$, i.e. $\\sin\\theta = 1.2197\\,\\lambda/D$.
"""
from __future__ import annotations
import numpy as np
from scipy.special import j1


def rect_pattern(theta, L, lam):
    """Normalised field pattern of a uniform line aperture."""
    return np.sinc(L * np.sin(theta) / lam)


def circ_pattern(theta, D, lam):
    """Normalised field pattern of a uniform circular aperture (Airy)."""
    u = np.pi * D / lam * np.sin(np.asarray(theta, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(u) < 1e-12, 1.0, 2 * j1(u) / u)
    return out


def directivity_rect(Lx, Ly, lam, eff=1.0):
    """$G = \\eta\\,4\\pi A/\\lambda^2$ for a planar aperture of area $A = L_xL_y$."""
    return eff * 4 * np.pi * Lx * Ly / lam ** 2


def far_field_fft(aperture, dx, lam, n_fft=1 << 14):
    """Far-field pattern of a sampled 1-D aperture illumination by FFT.
    Returns (sin(theta), normalised |E|). Spatial frequency $k_x/2\\pi =
    \\sin\\theta/\\lambda$."""
    E = np.fft.fftshift(np.fft.fft(aperture, n_fft))
    fx = np.fft.fftshift(np.fft.fftfreq(n_fft, dx))
    return fx * lam, np.abs(E) / np.abs(E).max()


def hpbw(pattern_fn, lam, size, lo=0.0, hi=None):
    """Numerically solve $|E(\\theta)|^2 = 1/2$ for the full half-power beamwidth [rad]."""
    from scipy.optimize import brentq
    hi = hi or lam / size
    return 2 * brentq(lambda t: pattern_fn(t, size, lam) ** 2 - 0.5, lo + 1e-12, hi)
