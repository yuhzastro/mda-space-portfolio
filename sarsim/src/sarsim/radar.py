"""Radar equation and SAR sensitivity.

Monostatic radar equation (from the Poynting flux of a spherical wave, PHYS 301):
$P_r = \\frac{P_t G^2 \\lambda^2 \\sigma}{(4\\pi)^3 R^4 L}$; thermal noise power
$N = k_B T_{sys} B$ (Johnson-Nyquist, PHYS 403).

Coherent processing gain. Range compression collects the pulse energy
$P_t\\tau$, and azimuth compression integrates $N_a = \\mathrm{PRF}\\,T_a$ pulses
over the synthetic aperture time $T_a = \\lambda R / (L_a V)$, so the image SNR of
a point target is

$\\mathrm{SNR} = \\frac{P_{avg} G^2 \\lambda^2 \\sigma\\,T_a}{(4\\pi)^3 R^4 k_B T F L},
\\quad P_{avg} = P_t\\tau\\,\\mathrm{PRF}.$

For a distributed target, $\\sigma = \\sigma^0 \\delta_a \\delta_{gr}$ with
$\\delta_a = L_a/2$ and $\\delta_{gr} = c/(2B\\sin\\theta_i)$. Setting SNR = 1 gives the
noise-equivalent sigma zero

$\\mathrm{NESZ} = \\frac{4(4\\pi)^3 R^3 V k_B T F L B \\sin\\theta_i}{P_{avg} G^2 \\lambda^3 c}.$
"""
from __future__ import annotations
import numpy as np

C = 299_792_458.0
K_B = 1.380649e-23


def received_power(Pt, G, lam, sigma, R, L=1.0):
    return Pt * G ** 2 * lam ** 2 * sigma / ((4 * np.pi) ** 3 * R ** 4 * L)


def point_target_snr(P_avg, G, lam, sigma, R, V, La, T_sys, F=1.0, L=1.0):
    Ta = lam * R / (La * V)
    return P_avg * G ** 2 * lam ** 2 * sigma * Ta / ((4 * np.pi) ** 3 * R ** 4 * K_B * T_sys * F * L)


def nesz(P_avg, G, lam, R, V, B, inc, T_sys, F=1.0, L=1.0):
    """Noise-equivalent sigma zero (linear)."""
    return 4 * (4 * np.pi) ** 3 * R ** 3 * V * K_B * T_sys * F * L * B * np.sin(inc) / (P_avg * G ** 2 * lam ** 3 * C)


def resolution(B, La, inc=None):
    """Slant-range ($0.886\\,c/2B$ for an unweighted spectrum) and nominal azimuth ($L_a/2$) resolution."""
    dr = 0.886 * C / (2 * B)
    return (dr if inc is None else dr / np.sin(inc)), La / 2


def ambiguity_prf_limits(V, La, swath_slant, duty=0.0):
    """Azimuth sampling needs PRF > 2V/L_a (Doppler bandwidth); unambiguous
    range needs PRF < c/(2 W_s) for slant swath $W_s$ (ignoring pulse duration)."""
    return 2 * V / La, C / (2 * swath_slant * (1 + duty))
