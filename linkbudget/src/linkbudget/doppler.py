"""Doppler shift on a satellite link: $f_D = -\\frac{f_0}{c}\\dot\\rho$ and its
rate $\\dot f_D = -\\frac{f_0}{c}\\ddot\\rho$ (first order in $\\dot\\rho/c$)."""
from __future__ import annotations
from datetime import timedelta
import numpy as np

C_KMS = 299792.458


def doppler_profile(sat, gs, t0, t1, f0_hz, step_s=1.0):
    """Time [s from t0], elevation [deg], range [km], Doppler [Hz], Doppler rate [Hz/s]."""
    n = int((t1 - t0).total_seconds() / step_s) + 1
    ts = np.arange(n) * step_s
    look = np.array([gs.look(sat, t0 + timedelta(seconds=float(t))) for t in ts])
    el, rng, rdot = np.degrees(look[:, 1]), look[:, 2], look[:, 3]
    fd = -f0_hz * rdot / C_KMS
    return ts, el, rng, fd, np.gradient(fd, ts)
