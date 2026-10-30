"""Relative motion for rendezvous and proximity operations (RPO).

Hill / LVLH frame of a circular chief orbit, mean motion $n$:
$x$ radial (out), $y$ along-track, $z$ orbit normal. Clohessy-Wiltshire (1960):

$\\ddot x - 2n\\dot y - 3n^2 x = 0,\\quad \\ddot y + 2n\\dot x = 0,\\quad \\ddot z + n^2 z = 0.$

ISS convention note: NASA's LVLH uses +x along velocity (V-bar), +z toward
Earth (R-bar). Convert with $x_{ISS} = y$, $z_{ISS} = -x$.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .constants import MU_EARTH


def cw_stm(n: float, t: float) -> np.ndarray:
    """Closed-form 6x6 state transition matrix $\\Phi(t)$ for $[x,y,z,\\dot x,\\dot y,\\dot z]$."""
    s, c = np.sin(n * t), np.cos(n * t)
    return np.array([
        [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
        [6 * (s - n * t), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * n * t) / n, 0],
        [0, 0, c, 0, 0, s / n],
        [3 * n * s, 0, 0, c, 2 * s, 0],
        [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
        [0, 0, -n * s, 0, 0, c]])


def cw_matrix(n: float) -> np.ndarray:
    """System matrix $A$ with $\\dot X = A X$."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 0], A[3, 4] = 3 * n * n, 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n * n
    return A


def two_impulse_transfer(n, r0, v0, rf, tf):
    """CW targeting: impulses $\\Delta v_1, \\Delta v_2$ taking $r_0 \\to r_f$ in $t_f$
    and nulling velocity at arrival (rendezvous at a hold point)."""
    P = cw_stm(n, tf)
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = P[:3, :3], P[:3, 3:], P[3:, :3], P[3:, 3:]
    v0_req = np.linalg.solve(Phi_rv, np.asarray(rf) - Phi_rr @ r0)
    vf = Phi_vr @ r0 + Phi_vv @ v0_req
    return v0_req - np.asarray(v0), -vf


def drift_free_vy(n, x0):
    """Along-track velocity giving a closed (drift-free) relative orbit:
    $\\dot y_0 = -2 n x_0$ (removes the secular $-3nt$ term)."""
    return -2.0 * n * x0


# --- nonlinear truth model --------------------------------------------------

def inertial_to_hill(rc, vc, rd, vd):
    """Deputy state relative to chief, expressed in the rotating Hill frame."""
    xh = rc / np.linalg.norm(rc)
    h = np.cross(rc, vc)
    zh = h / np.linalg.norm(h)
    yh = np.cross(zh, xh)
    Q = np.vstack([xh, yh, zh])
    w = np.array([0.0, 0.0, np.linalg.norm(h) / np.dot(rc, rc)])
    rho = Q @ (rd - rc)
    return rho, Q @ (vd - vc) - np.cross(w, rho)


def hill_to_inertial(rc, vc, rho, rhod):
    xh = rc / np.linalg.norm(rc)
    h = np.cross(rc, vc)
    zh = h / np.linalg.norm(h)
    yh = np.cross(zh, xh)
    Q = np.vstack([xh, yh, zh])
    w = np.array([0.0, 0.0, np.linalg.norm(h) / np.dot(rc, rc)])
    return rc + Q.T @ rho, vc + Q.T @ (rhod + np.cross(w, rho))


def propagate_nonlinear(rc0, vc0, rho0, rhod0, t_eval, mu=MU_EARTH):
    """Integrate both bodies under exact two-body gravity, return Hill-frame
    relative states at t_eval (shape (N, 6))."""
    rd0, vd0 = hill_to_inertial(rc0, vc0, rho0, rhod0)

    def f(_, y):
        out = np.empty(12)
        for k in (0, 6):
            r = y[k:k + 3]
            out[k:k + 3] = y[k + 3:k + 6]
            out[k + 3:k + 6] = -mu * r / np.linalg.norm(r) ** 3
        return out
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), np.concatenate([rc0, vc0, rd0, vd0]),
                    t_eval=t_eval, method="DOP853", rtol=1e-12, atol=1e-12)
    Y = sol.y.T
    return np.array([np.concatenate(inertial_to_hill(y[:3], y[3:6], y[6:9], y[9:12])) for y in Y])
