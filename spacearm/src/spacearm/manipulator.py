"""Serial manipulator kinematics with standard Denavit-Hartenberg parameters.

Link transform (standard DH, Spong et al. *Robot Modeling and Control*):
$^{i-1}T_i = R_z(\\theta_i)\\,T_z(d_i)\\,T_x(a_i)\\,R_x(\\alpha_i)$.

Geometric Jacobian for revolute joints, $J_i = [z_{i-1}\\times(p_e - p_{i-1});\\ z_{i-1}]$,
so that $[v_e;\\ \\omega_e] = J(q)\\,\\dot q$ in the base frame.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .so3 import log_so3


@dataclass
class DHLink:
    a: float        # m
    alpha: float    # rad
    d: float        # m
    theta_offset: float = 0.0
    q_min: float = -np.inf
    q_max: float = np.inf


def dh_transform(a, alpha, d, theta) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


@dataclass
class Manipulator:
    links: list[DHLink]
    qdot_max: np.ndarray | None = None   # rad/s per joint
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.links)

    def frames(self, q) -> list[np.ndarray]:
        """Base-frame transforms $^0T_0, ^0T_1, \\dots, ^0T_n$."""
        T = np.eye(4)
        out = [T]
        for L, qi in zip(self.links, q):
            T = T @ dh_transform(L.a, L.alpha, L.d, qi + L.theta_offset)
            out.append(T)
        return out

    def fk(self, q) -> np.ndarray:
        return self.frames(q)[-1]

    def jacobian(self, q) -> np.ndarray:
        Ts = self.frames(q)
        pe = Ts[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            z, p = Ts[i][:3, 2], Ts[i][:3, 3]
            J[:3, i] = np.cross(z, pe - p)
            J[3:, i] = z
        return J

    def manipulability(self, q) -> float:
        """Yoshikawa measure $w = \\sqrt{\\det(J J^T)}$."""
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def limits(self):
        return (np.array([L.q_min for L in self.links]), np.array([L.q_max for L in self.links]))


def pose_error(T, T_des) -> np.ndarray:
    """6-vector $[p_d - p;\\ \\log(R_d R^T)]$ in the base frame."""
    return np.concatenate([T_des[:3, 3] - T[:3, 3], log_so3(T_des[:3, :3] @ T[:3, :3].T)])


def dls_step(J, e, lam=0.05, W=None, z_null=None):
    """Damped least squares (Wampler 1986; Nakamura & Hanafusa 1986):
    $\\dot q = J^T (J J^T + \\lambda^2 I)^{-1} e + (I - J^+ J) z$.

    The damping bounds $\\|\\dot q\\|$ near singularities at the cost of a
    tracking error; the null-space term exploits redundancy (n > 6) without
    disturbing the end effector, to first order.
    """
    m = J.shape[0]
    dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(m), e)
    if z_null is not None:
        Jp = np.linalg.pinv(J)
        dq += (np.eye(J.shape[1]) - Jp @ J) @ z_null
    return dq


def joint_limit_gradient(q, q_min, q_max):
    """Negative gradient of the Liegeois (1977) cost
    $H(q) = \\frac{1}{2n}\\sum_i \\left(\\frac{q_i - \\bar q_i}{q_{i,max}-q_{i,min}}\\right)^2$."""
    mid, rng = (q_min + q_max) / 2, (q_max - q_min)
    return -(q - mid) / (len(q) * rng ** 2)


def solve_ik(arm: Manipulator, T_des, q0, tol_pos=1e-6, tol_rot=1e-6, max_iter=500,
             lam=0.05, null_gain=0.0):
    """Iterative IK by damped least squares with Levenberg-Marquardt damping
    adaptation: a step that does not reduce $\\|e\\|$ is rejected and
    $\\lambda$ is increased; an accepted step decreases it.
    Returns (q, converged, iterations)."""
    q = np.array(q0, float)
    qmin, qmax = arm.limits()
    finite = np.all(np.isfinite(qmin)) and np.all(np.isfinite(qmax))
    e = pose_error(arm.fk(q), T_des)
    for k in range(max_iter):
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            return q, True, k
        z = null_gain * joint_limit_gradient(q, qmin, qmax) if (null_gain and finite) else None
        q_new = np.clip(q + dls_step(arm.jacobian(q), e, lam, z_null=z), qmin, qmax)
        e_new = pose_error(arm.fk(q_new), T_des)
        if np.linalg.norm(e_new) < np.linalg.norm(e):
            q, e, lam = q_new, e_new, max(lam / 2.0, 1e-6)
        else:
            lam *= 4.0
    return q, False, max_iter


def ssrms_like() -> Manipulator:
    """A 7-DOF, symmetric, SSRMS-like (Canadarm2) arm.

    Joint sequence: shoulder roll, yaw, pitch; elbow pitch; wrist pitch, yaw,
    roll. Two 6.85 m booms, giving the 17.6 m class reach of SSRMS. Joint
    limits are +/-270 deg and joint rate limits 4 deg/s.

    **Approximate geometry, not flight data.** The offsets are representative
    values chosen to keep SSRMS's symmetric, offset-joint structure, which has
    no spherical wrist and so needs a numerical IK solution.
    """
    lim = np.radians(270.0)
    h = np.pi / 2
    links = [DHLink(0.0, h, 0.380, 0, -lim, lim),     # shoulder roll
             DHLink(0.0, h, 0.635, h, -lim, lim),     # shoulder yaw
             DHLink(6.850, 0.0, 0.504, 0, -lim, lim), # shoulder pitch + upper boom
             DHLink(6.850, 0.0, 0.000, 0, -lim, lim), # elbow pitch + lower boom
             DHLink(0.0, h, 0.504, 0, -lim, lim),     # wrist pitch
             DHLink(0.0, h, 0.635, 0, -lim, lim),     # wrist yaw
             DHLink(0.0, 0.0, 0.380, 0, -lim, lim)]   # wrist roll
    return Manipulator(links, qdot_max=np.full(7, np.radians(4.0)), name="SSRMS-like 7-DOF")
