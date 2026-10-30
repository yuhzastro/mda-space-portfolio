"""Free-flyer capture by a station-mounted arm (berthing-style capture).

Scenario modelled on ISS cargo-vehicle capture (e.g. HTV, Cygnus): the
vehicle climbs the R-bar to a hold point about 10 m below the station, goes
to free drift, and the arm grapples its fixture. The arm then tracks the
drifting fixture with resolved-rate control.

Assumptions (stated, not hidden):
 * Station attitude holds LVLH, so the arm base is fixed in the Hill frame.
 * Station mass >> vehicle mass: fixed-base arm, no reaction on the station.
 * Vehicle translational drift follows CW; attitude drifts at a constant
   body rate.
 * Capture envelope (5 cm, 2 deg) is an assumed value for the demo.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .manipulator import Manipulator, pose_error, dls_step
from .proximity import cw_stm
from .so3 import exp_so3, rot_y


@dataclass
class CaptureConfig:
    n: float = 2 * np.pi / 5554.0                  # ISS-like mean motion [rad/s]
    base_pos: np.ndarray = field(default_factory=lambda: np.array([-2.0, 0.0, 0.0]))
    base_R: np.ndarray = field(default_factory=lambda: rot_y(-np.pi / 2))  # base z axis = nadir (-x Hill)
    vehicle_x0: np.ndarray = field(default_factory=lambda: np.array(
        [-12.0, 0.0, 0.0, 0.0, 0.0, 0.0]))         # Hill state [m, m/s]
    vehicle_rate: np.ndarray = field(default_factory=lambda: np.radians([0.0, 0.0, 0.02]))  # rad/s, body
    gf_offset: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.0, 0.0]))       # fixture, vehicle body
    standoff: float = 1.0          # m, pre-grasp distance along fixture axis
    v_tip_max: float = 0.10        # m/s commanded tip speed limit
    k_p: float = 0.3               # 1/s position/attitude gain
    lam: float = 0.08              # DLS damping
    dt: float = 0.1                # s
    t_max: float = 900.0           # s
    tol_pos: float = 0.05          # m, capture envelope (assumed)
    tol_rot: float = np.radians(2.0)


def vehicle_state(cfg: CaptureConfig, t: float):
    """Vehicle position, velocity, attitude and angular velocity in Hill frame."""
    X = cw_stm(cfg.n, t) @ cfg.vehicle_x0
    R = exp_so3(cfg.vehicle_rate * t)
    return X[:3], X[3:], R, R @ cfg.vehicle_rate


def fixture_pose(cfg: CaptureConfig, t: float, standoff: float = 0.0):
    """Grasp pose in the arm base frame, and its twist (feed-forward).

    The end-effector z axis points into the fixture, i.e. along -x_body.
    """
    p, v, R, w = vehicle_state(cfg, t)
    r_gf = R @ cfg.gf_offset
    approach = -R[:, 0]                              # EE z axis in Hill frame
    R_ee = R @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)  # columns: x, y, z(=-x_body)
    p_ee = p + r_gf - standoff * approach
    v_ee = v + np.cross(w, p_ee - p)
    Bt = cfg.base_R.T
    T = np.eye(4)
    T[:3, :3] = Bt @ R_ee
    T[:3, 3] = Bt @ (p_ee - cfg.base_pos)
    return T, np.concatenate([Bt @ v_ee, Bt @ w])


@dataclass
class CaptureResult:
    t: np.ndarray
    q: np.ndarray
    pos_err: np.ndarray
    rot_err: np.ndarray
    phase: np.ndarray
    captured: bool
    t_capture: float | None
    manipulability: np.ndarray


def saturate_joint_rates(dq, qdot_max):
    """Uniform scaling keeps the Cartesian direction of motion."""
    s = np.max(np.abs(dq) / qdot_max)
    return dq / s if s > 1.0 else dq


def simulate_capture(arm: Manipulator, q0, cfg: CaptureConfig | None = None) -> CaptureResult:
    cfg = cfg or CaptureConfig()
    q = np.array(q0, float)
    qmin, qmax = arm.limits()
    ts, qs, pe, re, ph, mw = [], [], [], [], [], []
    phase, t_cap = 0, None          # 0: approach to pre-grasp, 1: final approach
    t = 0.0
    while t <= cfg.t_max:
        T_des, twist = fixture_pose(cfg, t, cfg.standoff if phase == 0 else 0.0)
        e = pose_error(arm.fk(q), T_des)
        ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        ts.append(t); qs.append(q.copy()); pe.append(ep); re.append(er); ph.append(phase)
        mw.append(arm.manipulability(q))
        if phase == 0 and ep < 0.02 and er < np.radians(1.0):
            phase = 1
        elif phase == 1 and ep < cfg.tol_pos and er < cfg.tol_rot:
            t_cap = t
            break
        xdot = twist + cfg.k_p * e
        vn = np.linalg.norm(xdot[:3])
        if vn > cfg.v_tip_max:
            xdot[:3] *= cfg.v_tip_max / vn
        dq = saturate_joint_rates(dls_step(arm.jacobian(q), xdot, cfg.lam), arm.qdot_max)
        q = np.clip(q + dq * cfg.dt, qmin, qmax)
        t += cfg.dt
    return CaptureResult(np.array(ts), np.array(qs), np.array(pe), np.array(re), np.array(ph),
                         t_cap is not None, t_cap, np.array(mw))
