"""Batch least-squares orbit determination from ground-station tracking.

Measurements: range $\\rho = |r - R_s|$ and range rate
$\\dot\\rho = \\frac{(r - R_s)\\cdot(v - \\omega_\\oplus\\times R_s)}{\\rho}$ from
stations fixed on a rotating Earth. The state $x_0 = (r_0, v_0)$ at epoch
is estimated by Gauss-Newton iterations on the weighted normal equations
(PHYS 229 data analysis):

$\\delta x = (H^TWH)^{-1}H^TW\\,\\delta y$, $H = \\partial y/\\partial x_0$, $W = \\mathrm{diag}(1/\\sigma_i^2)$,

with covariance $P = (H^TWH)^{-1}$. For a correct noise model the weighted
residual sum $\\chi^2 = \\delta y^TW\\delta y$ follows a $\\chi^2_{m-6}$ distribution.
$H$ is computed by central differences through the J2 dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .constants import OMEGA_EARTH
from .frames import geodetic_to_ecef
from .j2 import propagate_cowell
from .frames import rot3


@dataclass
class Station:
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_el_deg: float = 5.0

    def eci(self, t, theta0=0.0):
        """Inertial position/velocity, Earth angle theta0 + omega t."""
        r = rot3(-(theta0 + OMEGA_EARTH * t)) @ geodetic_to_ecef(np.radians(self.lat_deg), np.radians(self.lon_deg), self.alt_km)
        return r, np.cross([0, 0, OMEGA_EARTH], r)


def measure(states, times, stations, theta0=0.0):
    """Noise-free (range, range-rate) for every station/time with the satellite above the mask.
    Returns list of (time index, station index, rho, rhodot)."""
    out = []
    for k, (t, x) in enumerate(zip(times, states)):
        for j, st in enumerate(stations):
            rs, vs = st.eci(t, theta0)
            d = x[:3] - rs
            rho = np.linalg.norm(d)
            up = rs / np.linalg.norm(rs)
            if np.dot(d, up) / rho < np.sin(np.radians(st.min_el_deg)):
                continue
            out.append((k, j, rho, np.dot(d, x[3:] - vs) / rho))
    return out


def _predict(x0, times, obs, stations, theta0):
    _, R, V = propagate_cowell(x0[:3], x0[3:], (0, times[-1]), t_eval=times)
    X = np.hstack([R, V])
    y = []
    for k, j, _, _ in obs:
        rs, vs = stations[j].eci(times[k], theta0)
        d = X[k, :3] - rs
        rho = np.linalg.norm(d)
        y += [rho, np.dot(d, X[k, 3:] - vs) / rho]
    return np.array(y)


@dataclass
class ODResult:
    x0: np.ndarray
    P: np.ndarray
    chi2: float
    dof: int
    iterations: int
    residuals: np.ndarray


def batch_least_squares(x_guess, times, obs, stations, sig_rho, sig_rhodot, theta0=0.0,
                        max_iter=10, tol=1e-6):
    y = np.ravel([[o[2], o[3]] for o in obs])
    w = np.tile([1 / sig_rho ** 2, 1 / sig_rhodot ** 2], len(obs))
    x = np.array(x_guess, float)
    steps = np.array([1e-3] * 3 + [1e-6] * 3)
    for it in range(1, max_iter + 1):
        yp = _predict(x, times, obs, stations, theta0)
        H = np.empty((len(y), 6))
        for i in range(6):
            dx = np.zeros(6); dx[i] = steps[i]
            H[:, i] = (_predict(x + dx, times, obs, stations, theta0)
                       - _predict(x - dx, times, obs, stations, theta0)) / (2 * steps[i])
        N = H.T @ (w[:, None] * H)
        delta = np.linalg.solve(N, H.T @ (w * (y - yp)))
        x += delta
        if np.linalg.norm(delta[:3]) < tol and np.linalg.norm(delta[3:]) < tol * 1e-3:
            break
    res = y - _predict(x, times, obs, stations, theta0)
    return ODResult(x, np.linalg.inv(N), float(res @ (w * res)), len(y) - 6, it, res)
