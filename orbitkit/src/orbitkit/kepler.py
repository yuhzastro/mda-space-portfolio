"""Two-body orbital elements <-> state vectors (Vallado algorithms 9 and 10)."""
from __future__ import annotations
import numpy as np
from .constants import MU_EARTH


def solve_kepler(M: float, e: float, tol: float = 1e-14) -> float:
    """Solve $M = E - e\\sin E$ for eccentric anomaly by Newton-Raphson."""
    E = M if e < 0.8 else np.pi
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def true_from_mean(M: float, e: float) -> float:
    E = solve_kepler(M, e)
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def coe2rv(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """Classical elements (km, -, rad...) -> inertial r [km], v [km/s]."""
    p = a * (1.0 - e * e)
    r_pqw = np.array([p * np.cos(nu), p * np.sin(nu), 0.0]) / (1.0 + e * np.cos(nu))
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = np.cos(raan), np.sin(raan), np.cos(i), np.sin(i), np.cos(argp), np.sin(argp)
    Q = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return Q @ r_pqw, Q @ v_pqw


def rv2coe(r, v, mu=MU_EARTH):
    """Inertial state -> (a, e, i, raan, argp, nu). Non-circular, non-equatorial."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    n = np.cross([0, 0, 1.0], h)
    evec = ((vn ** 2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - vn ** 2 / mu)
    i = np.arccos(h[2] / np.linalg.norm(h))
    raan = np.arccos(n[0] / np.linalg.norm(n))
    if n[1] < 0:
        raan = 2 * np.pi - raan
    argp = np.arccos(np.clip(np.dot(n, evec) / (np.linalg.norm(n) * e), -1, 1))
    if evec[2] < 0:
        argp = 2 * np.pi - argp
    nu = np.arccos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
    if np.dot(r, v) < 0:
        nu = 2 * np.pi - nu
    return a, e, i, raan, argp, nu
