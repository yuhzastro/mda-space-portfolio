"""Demo: star-tracker attitude determination, then a 3-axis quaternion PD slew,
plus the intermediate-axis (Dzhanibekov) instability of a free rigid body."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spacearm.attitude import (BRIGHT_STARS, radec_to_unit, q_to_dcm, davenport_q, triad, dcm_to_q, q_angle,
                               propagate_attitude, quaternion_pd_torque, star_photon_rate)

rng = np.random.default_rng(3)
q_true = np.array([0.8, 0.2, -0.4, 0.4]); q_true /= np.linalg.norm(q_true)
names = list(BRIGHT_STARS)[:8]
rs = [radec_to_unit(*BRIGHT_STARS[n][:2]) for n in names]
# per-star noise: shot-noise centroiding, sigma ~ (PSF width) / sqrt(N_photons), for a 25 mm aperture
# and 0.1 s exposure, added in quadrature to a 3 arcsec floor (pixel response, calibration, optics
# distortion), which dominates for bright stars in real trackers
psf, floor = np.radians(30 / 3600), np.radians(3 / 3600)
sig = [np.hypot(psf / np.sqrt(star_photon_rate(BRIGHT_STARS[n][2], np.pi * 0.0125 ** 2) * 0.1), floor) for n in names]
bs = [q_to_dcm(q_true).T @ r + s * rng.normal(size=3) for r, s in zip(rs, sig)]
bs = [b / np.linalg.norm(b) for b in bs]
q_est = davenport_q(bs, rs, w=1 / np.array(sig) ** 2)
q_tri = dcm_to_q(triad(bs[0], bs[1], rs[0], rs[1]))
print("star noise [arcsec]:", ", ".join(f"{n} {np.degrees(s)*3600:.2f}" for n, s in zip(names, sig)))
print(f"q-method error {np.degrees(q_angle(q_est, q_true))*3600:.2f} arcsec, TRIAD error {np.degrees(q_angle(q_tri, q_true))*3600:.2f} arcsec")

I = np.diag([120.0, 100.0, 80.0])
t, Q, W = propagate_attitude(I, q_est, np.zeros(3), 400.0, torque=quaternion_pd_torque(np.array([1.0, 0, 0, 0]), 4.0, 40.0),
                             t_eval=np.linspace(0, 400, 801))
err = [np.degrees(q_angle(q, [1, 0, 0, 0])) for q in Q]
print(f"slew of {err[0]:.1f} deg settles below 0.1 deg after {t[np.argmax(np.array(err) < 0.1)]:.0f} s")

Id = np.diag([10.0, 15.0, 20.0])
t2, Q2, W2 = propagate_attitude(Id, [1, 0, 0, 0], [1e-3, 1.0, 1e-3], 150.0, t_eval=np.linspace(0, 150, 3000))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].semilogy(t, err); ax[0].set(xlabel="t [s]", ylabel="attitude error [deg]", title="Quaternion PD slew"); ax[0].grid(alpha=.3)
for k, lab in enumerate(["$\\omega_1$", "$\\omega_2$ (intermediate axis)", "$\\omega_3$"]):
    ax[1].plot(t2, W2[:, k], label=lab)
ax[1].set(xlabel="t [s]", ylabel="body rate [rad/s]", title="Intermediate-axis instability"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("examples/attitude_demo.png", dpi=120)
print("saved examples/attitude_demo.png")
