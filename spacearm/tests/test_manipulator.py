import numpy as np
import pytest
from spacearm.manipulator import (Manipulator, DHLink, ssrms_like, solve_ik, dls_step,
                                  pose_error, dh_transform)
from spacearm.so3 import exp_so3, log_so3

rng = np.random.default_rng(42)


def test_planar_2r_closed_form():
    """2R planar arm: $x = l_1 c_1 + l_2 c_{12}$, $y = l_1 s_1 + l_2 s_{12}$."""
    arm = Manipulator([DHLink(1.0, 0, 0), DHLink(0.7, 0, 0)])
    q = np.array([0.4, -1.1])
    p = arm.fk(q)[:3, 3]
    assert np.allclose(p[:2], [np.cos(.4) + .7 * np.cos(-.7), np.sin(.4) + .7 * np.sin(-.7)])
    # singular when fully stretched: det J_xy = l1 l2 sin q2
    J = arm.jacobian(q)[:2]
    assert np.linalg.det(J) == pytest.approx(1.0 * 0.7 * np.sin(-1.1))


def test_dh_transform_is_rigid():
    T = dh_transform(0.3, 0.7, -0.2, 1.3)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_jacobian_matches_finite_difference(seed):
    arm = ssrms_like()
    q = np.random.default_rng(seed).uniform(-np.pi, np.pi, 7)
    J = arm.jacobian(q)
    h = 1e-6
    T0 = arm.fk(q)
    for i in range(7):
        dq = np.zeros(7); dq[i] = h
        Tp, Tm = arm.fk(q + dq), arm.fk(q - dq)
        v_fd = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        w_fd = log_so3(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
        assert np.allclose(J[:3, i], v_fd, atol=1e-6)
        assert np.allclose(J[3:, i], w_fd, atol=1e-6)


def test_ssrms_reach():
    arm = ssrms_like()
    # fully extended booms: distance between shoulder-pitch and wrist-pitch axes = 13.7 m
    reach = max(np.linalg.norm(arm.fk(q)[:3, 3]) for q in rng.uniform(-np.pi, np.pi, (3000, 7)))
    assert 14.0 < reach < 17.6


def test_so3_log_exp_inverse():
    for _ in range(50):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        assert np.allclose(log_so3(exp_so3(w)), w, atol=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_ik_recovers_reachable_poses(seed):
    arm = ssrms_like()
    r = np.random.default_rng(100 + seed)
    q_true = r.uniform(-2.0, 2.0, 7)
    T_des = arm.fk(q_true)
    q0 = q_true + r.normal(scale=0.3, size=7)
    q, ok, _ = solve_ik(arm, T_des, q0, lam=0.01, max_iter=1000)
    assert ok
    assert np.linalg.norm(pose_error(arm.fk(q), T_des)) < 1e-5


def test_null_space_motion_does_not_move_end_effector():
    """For a 7-DOF arm, $(I - J^+J)z$ is a self-motion: $J (I - J^+J) z = 0$."""
    arm = ssrms_like()
    q = np.radians([10, 30, -60, 100, -40, 20, 5])
    J = arm.jacobian(q)
    dq = dls_step(J, np.zeros(6), lam=0.0, z_null=rng.normal(size=7))
    assert np.linalg.norm(J @ dq) < 1e-10
    assert np.linalg.norm(dq) > 1e-3


def test_dls_bounded_at_singularity():
    """At a singular configuration the pseudo-inverse blows up; DLS keeps
    $\\|\\dot q\\| \\le \\|e\\| / (2\\lambda)$."""
    arm = Manipulator([DHLink(1.0, 0, 0), DHLink(1.0, 0, 0)])
    J = arm.jacobian([0.3, 0.0])[:2]          # stretched: rank 1
    e = np.array([np.cos(.3), np.sin(.3)])    # push outward, unreachable direction
    lam = 0.1
    dq = dls_step(J, e, lam)
    assert np.linalg.norm(dq) <= np.linalg.norm(e) / (2 * lam) + 1e-12
