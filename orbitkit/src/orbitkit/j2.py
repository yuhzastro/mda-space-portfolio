"""J2 perturbation: analytic secular rates and numerical (Cowell) propagation.

Secular rates (Vallado eq. 9-41, first order in J2), with $p = a(1-e^2)$ and
$n = \\sqrt{\\mu/a^3}$:

$\\dot\\Omega = -\\tfrac{3}{2} n J_2 (R_\\oplus/p)^2 \\cos i$

$\\dot\\omega = \\tfrac{3}{4} n J_2 (R_\\oplus/p)^2 (4 - 5\\sin^2 i)$

$\\dot M = n\\left[1 + \\tfrac{3}{4} J_2 (R_\\oplus/p)^2 \\sqrt{1-e^2}(2 - 3\\sin^2 i)\\right]$
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .constants import MU_EARTH, R_EARTH, J2, SIDEREAL_DAY


def secular_rates(a, e, i, mu=MU_EARTH):
    """Return $(\\dot\\Omega, \\dot\\omega, \\dot M)$ in rad/s."""
    n = np.sqrt(mu / a ** 3)
    k = J2 * (R_EARTH / (a * (1 - e * e))) ** 2
    si2 = np.sin(i) ** 2
    raan_dot = -1.5 * n * k * np.cos(i)
    argp_dot = 0.75 * n * k * (4.0 - 5.0 * si2)
    m_dot = n * (1.0 + 0.75 * k * np.sqrt(1 - e * e) * (2.0 - 3.0 * si2))
    return raan_dot, argp_dot, m_dot


def sun_synchronous_inclination(a, e=0.0):
    """Inclination [rad] giving $\\dot\\Omega = 2\\pi$ per tropical year
    (0.98565 deg/day), so the orbit plane tracks the mean Sun."""
    target = 2 * np.pi / (365.2421897 * 86400.0)
    n = np.sqrt(MU_EARTH / a ** 3)
    k = J2 * (R_EARTH / (a * (1 - e * e))) ** 2
    return float(np.arccos(-target / (1.5 * n * k)))


def accel_j2(r, mu=MU_EARTH):
    """Two-body + J2 acceleration [km/s^2] (gradient of the J2 potential)."""
    x, y, z = r
    rn2 = x * x + y * y + z * z
    rn = np.sqrt(rn2)
    f = 1.5 * J2 * mu * R_EARTH ** 2 / rn ** 5
    zr = 5.0 * z * z / rn2
    return -mu * r / rn ** 3 + f * np.array([x * (zr - 1), y * (zr - 1), z * (zr - 3)])


def propagate_cowell(r0, v0, t_span, t_eval=None, j2=True, rtol=1e-12, atol=1e-12):
    """Numerically integrate $\\ddot r = -\\mu r/|r|^3 + a_{J2}$ with DOP853."""
    def f(_, y):
        a = accel_j2(y[:3]) if j2 else -MU_EARTH * y[:3] / np.linalg.norm(y[:3]) ** 3
        return np.concatenate([y[3:], a])
    sol = solve_ivp(f, t_span, np.concatenate([r0, v0]), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    return sol.t, sol.y[:3].T, sol.y[3:].T


def potential_j2(r, mu=MU_EARTH):
    """Gravitational potential through $J_2$ [km^2/s^2], with $\\ddot r = -\\nabla U$:

    $U = -\\frac{\\mu}{r}\\left[1 - J_2\\left(\\frac{R_\\oplus}{r}\\right)^2 P_2(\\sin\\phi)\\right]$,
    $P_2(x) = \\tfrac{1}{2}(3x^2 - 1)$, $\\sin\\phi = z/r$.

    `accel_j2` is the analytic gradient of this scalar field (MATH 217); the
    tests check it against a finite-difference gradient.
    """
    r = np.asarray(r, float)
    rn = np.linalg.norm(r)
    s = r[2] / rn
    return -mu / rn * (1 - J2 * (R_EARTH / rn) ** 2 * 0.5 * (3 * s * s - 1))
