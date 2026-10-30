"""Demo: V-bar rendezvous to a hold point, R-bar hold, then arm capture.

1. CW two-impulse transfer from 2 km behind the station to the 250 m V-bar
   hold point, checked against a nonlinear two-body propagation.
2. SSRMS-like arm captures a free-drifting vehicle 10 m below the arm base.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spacearm.constants import MU_EARTH, R_EARTH
from spacearm.proximity import cw_stm, two_impulse_transfer, propagate_nonlinear
from spacearm.manipulator import ssrms_like
from spacearm.capture import CaptureConfig, simulate_capture

a = R_EARTH + 420e3
n = np.sqrt(MU_EARTH / a ** 3)
rc0 = np.array([a, 0, 0.0])
vc0 = np.sqrt(MU_EARTH / a) * np.array([0, np.cos(np.radians(51.6)), np.sin(np.radians(51.6))])

r0, rf, tf = np.array([0, -2000.0, 0]), np.array([0, -250.0, 0]), np.pi / n
dv1, dv2 = two_impulse_transfer(n, r0, np.zeros(3), rf, tf)
ts = np.linspace(0, tf, 300)
cw = np.array([cw_stm(n, t) @ np.concatenate([r0, dv1]) for t in ts])
nl = propagate_nonlinear(rc0, vc0, r0, dv1, ts)
print(f"Mean motion n = {n:.6e} rad/s, period = {2*np.pi/n/60:.2f} min")
print(f"dv1 = {dv1.round(4)} m/s, dv2 = {dv2.round(4)} m/s, total {np.linalg.norm(dv1)+np.linalg.norm(dv2):.4f} m/s")
print(f"CW vs nonlinear miss at hold point: {np.linalg.norm(nl[-1,:3]-rf):.3f} m")

arm = ssrms_like()
cfg = CaptureConfig(vehicle_x0=np.array([-12.0, 0.3, -0.2, 0.002, -0.003, 0.001]))
res = simulate_capture(arm, np.radians([0, 30, -60, 120, -60, 0, 0]), cfg)
print(f"Capture: {res.captured} at t = {res.t_capture:.1f} s, final error "
      f"{res.pos_err[-1]*100:.1f} cm / {np.degrees(res.rot_err[-1]):.2f} deg")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].plot(cw[:, 1], cw[:, 0], label="Clohessy-Wiltshire")
ax[0].plot(nl[:, 1], nl[:, 0], "--", label="Nonlinear two-body")
ax[0].plot(0, 0, "ks", label="Station"); ax[0].plot(*rf[[1, 0]], "r^", label="Hold point")
ax[0].set(xlabel="y, along-track (V-bar) [m]", ylabel="x, radial [m]", title="V-bar two-impulse transfer")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].semilogy(res.t, res.pos_err, label="position [m]")
ax[1].semilogy(res.t, np.degrees(res.rot_err), label="attitude [deg]")
ax[1].axhline(cfg.tol_pos, color="k", ls=":")
ax[1].set(xlabel="t [s]", title="Arm tracking error to grapple fixture"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(res.t, np.degrees(res.q)); ax[2].set(xlabel="t [s]", ylabel="joint angle [deg]",
                                                 title="SSRMS-like joint trajectories"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig("examples/demo.png", dpi=120)
print("saved examples/demo.png")
