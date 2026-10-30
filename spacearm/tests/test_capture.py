import numpy as np
import pytest
from spacearm.manipulator import ssrms_like
from spacearm.capture import CaptureConfig, simulate_capture, fixture_pose


@pytest.mark.parametrize("x0", [
    [-12.0, 0, 0, 0, 0, 0],                       # ideal free drift
    [-12.0, 0.3, -0.2, 0.002, -0.003, 0.001],     # mm/s residual drift and offset
])
def test_capture_succeeds_within_limits(x0):
    arm = ssrms_like()
    cfg = CaptureConfig(vehicle_x0=np.array(x0))
    res = simulate_capture(arm, np.radians([0, 30, -60, 120, -60, 0, 0]), cfg)
    assert res.captured and res.t_capture < 600
    # joint rate limit respected at every step
    dq = np.abs(np.diff(res.q, axis=0)) / cfg.dt
    assert np.all(dq <= arm.qdot_max + 1e-9)
    # stays away from singularities (manipulability bounded below)
    assert res.manipulability.min() > 1.0


def test_fixture_pose_is_rigid_transform():
    T, tw = fixture_pose(CaptureConfig(), 123.0)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
