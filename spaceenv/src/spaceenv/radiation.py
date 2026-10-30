"""Charged particles in Earth's dipole field: the radiation belts.

Dipole field (PHYS 301), magnetic latitude $\\lambda$, $r$ in Earth radii:
$B(r,\\lambda) = \\frac{B_0}{r^3}\\sqrt{1 + 3\\sin^2\\lambda}$; field line
$r = L\\cos^2\\lambda$.

Relativistic kinematics (PHYS 200): $\\gamma = 1 + W/(mc^2)$, $p = \\gamma m v$.
Gyrofrequency $\\Omega = |q|B/(\\gamma m)$, gyroradius $r_g = p_\\perp/(|q|B)$.

Adiabatic invariants: $\\mu = p_\\perp^2/(2mB)$ is conserved, so a particle with
equatorial pitch angle $\\alpha_{eq}$ mirrors where $B_m = B_{eq}/\\sin^2\\alpha_{eq}$, i.e.
$\\frac{\\cos^6\\lambda_m}{\\sqrt{1+3\\sin^2\\lambda_m}} = \\sin^2\\alpha_{eq}$.

Guiding-centre period approximations (Hamlin et al. 1961; Baumjohann & Treumann 1997):
bounce $\\tau_b \\approx \\frac{4LR_E}{v}(1.30 - 0.56\\sin\\alpha_{eq})$, drift
$\\tau_d \\approx \\frac{\\pi |q| B_0 R_E^2}{3 L\\,(\\gamma m v^2/2)}\\,\\frac{1}{0.35 + 0.15\\sin\\alpha_{eq}}$.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .constants import C, R_EARTH, B0_DIPOLE


def dipole_B(r_vec, B0=B0_DIPOLE, Re=R_EARTH):
    """Field vector [T] of a centred dipole aligned with -z (Earth-like), r in metres.
    $\\mathbf B = \\frac{B_0R_E^3}{r^5}\\left[3(\\hat m\\cdot\\mathbf r)\\mathbf r - r^2\\hat m\\right]$ with $\\hat m = -\\hat z$."""
    r_vec = np.asarray(r_vec, float)
    r2 = r_vec @ r_vec
    r = np.sqrt(r2)
    m = np.array([0.0, 0.0, -1.0])
    return B0 * Re ** 3 / r ** 5 * (3 * (m @ r_vec) * r_vec - r2 * m)


def B_mag(r_re, lat):
    return B0_DIPOLE / r_re ** 3 * np.sqrt(1 + 3 * np.sin(lat) ** 2)


def gamma_from_kinetic(W_J, m):
    return 1 + W_J / (m * C ** 2)


def speed(W_J, m):
    g = gamma_from_kinetic(W_J, m)
    return C * np.sqrt(1 - 1 / g ** 2)


def mirror_latitude(alpha_eq):
    """Magnetic latitude [rad] of the mirror point (independent of L)."""
    s2 = np.sin(alpha_eq) ** 2
    f = lambda lam: np.cos(lam) ** 6 / np.sqrt(1 + 3 * np.sin(lam) ** 2) - s2
    return brentq(f, 0.0, np.pi / 2 - 1e-9) if s2 < 1 else 0.0


def loss_cone(L, h_atm_m=100e3):
    """Equatorial loss-cone half angle: particles mirroring below altitude h are lost."""
    r_foot = (R_EARTH + h_atm_m) / R_EARTH
    lam = np.arccos(np.sqrt(r_foot / L))
    return np.arcsin(np.sqrt(B_mag(L, 0.0) / B_mag(r_foot, lam)))


def bounce_period(L, W_J, m, alpha_eq):
    return 4 * L * R_EARTH / speed(W_J, m) * (1.30 - 0.56 * np.sin(alpha_eq))


def drift_period(L, W_J, m, alpha_eq, q=1.602176634e-19):
    g, v = gamma_from_kinetic(W_J, m), speed(W_J, m)
    return np.pi * q * B0_DIPOLE * R_EARTH ** 2 / (3 * L * 0.5 * g * m * v ** 2) / (0.35 + 0.15 * np.sin(alpha_eq))


def stormer_vertical_cutoff(L_or_r, lat):
    """Vertical cutoff rigidity [V] for particles arriving from infinity
    (Stormer): $R_c = \\frac{cB_0R_E}{4}\\frac{\\cos^4\\lambda}{r^2}$ (r in Earth radii).
    Particles with $pc/|q|$ below this cannot reach that point."""
    return C * B0_DIPOLE * R_EARTH / 4 * np.cos(lat) ** 4 / L_or_r ** 2


def boris_push(x0, v0, q, m, t_end, dt, field=dipole_B, record_every=10):
    """Relativistic Boris integrator for $\\frac{d(\\gamma m\\mathbf v)}{dt} = q\\mathbf v\\times\\mathbf B$
    (E = 0). Volume-preserving, conserves energy exactly in a pure magnetic field.
    Returns times and positions/velocities sampled every `record_every` steps."""
    x = np.array(x0, float)
    u = np.array(v0, float) / np.sqrt(1 - (np.array(v0) @ np.array(v0)) / C ** 2)   # u = gamma v
    n = int(t_end / dt)
    ts, xs, vs = [], [], []
    for k in range(n):
        g = np.sqrt(1 + u @ u / C ** 2)
        x_half = x + 0.5 * dt * u / g
        tvec = q * dt / (2 * g * m) * field(x_half)
        svec = 2 * tvec / (1 + tvec @ tvec)
        up = u + np.cross(u, tvec)
        u = u + np.cross(up, svec)
        g = np.sqrt(1 + u @ u / C ** 2)
        x = x_half + 0.5 * dt * u / g
        if k % record_every == 0:
            ts.append((k + 1) * dt); xs.append(x.copy()); vs.append(u / g)
    return np.array(ts), np.array(xs), np.array(vs)
