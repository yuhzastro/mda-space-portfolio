"""Demo: focus a C-band stripmap scene of corner reflectors, RADARSAT-2-class
parameters (15 m antenna, 5.405 GHz, 30 MHz chirp), and print the system
performance numbers. Parameters are representative, not flight values."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sarsim.rda import SARParams, simulate_raw, rda, range_compress, C
from sarsim.radar import nesz, resolution, ambiguity_prf_limits
from sarsim.antenna import rect_pattern, circ_pattern
from sarsim.chirp import irf_metrics

p = SARParams()
dr = C / (2 * p.fs)
targets = [(p.range_axis()[p.n_rg // 2 + dj], di / p.prf, 1.0)
           for dj, di in [(0, 0), (-120, -150), (120, -150), (-120, 150), (120, 150), (0, 300)]]
rng = np.random.default_rng(7)
raw = simulate_raw(p, targets, noise_std=0.05, rng=rng)
rc = range_compress(p, raw)
img = rda(p, raw)

inc = np.radians(35.0)
s0 = nesz(300.0, 10 ** 4.5, p.lam, p.R_ref, p.Vr, p.B, inc, 600.0, 10 ** 0.3, 10 ** 0.5)
dsr, da = resolution(p.B, p.La)
prf_min, prf_max = ambiguity_prf_limits(p.Vr, p.La, 50e3 * np.sin(inc))
a = np.abs(img)
I, J = p.n_az // 2, p.n_rg // 2          # the centre reflector
w_r, pslr_r, islr_r = irf_metrics(img[I, J - 100:J + 100])
w_a, pslr_a, _ = irf_metrics(img[I - 128:I + 128, J])
print(f"wavelength {p.lam*100:.2f} cm, chirp TB = {p.Tp*p.B:.0f}")
print(f"slant-range resolution: theory {dsr:.2f} m, measured {w_r*dr:.2f} m, PSLR {pslr_r:.2f} dB, ISLR {islr_r:.2f} dB")
print(f"ground-range resolution at 35 deg incidence: {dsr/np.sin(inc):.2f} m")
print(f"azimuth resolution: L_a/2 = {da:.2f} m, measured {w_a*p.Vr/p.prf:.2f} m")
print(f"NESZ (300 W avg, 45 dBi, 600 K, F = 3 dB, L = 5 dB): {10*np.log10(s0):.1f} dB")
print(f"PRF window for a 50 km ground swath: {prf_min:.0f} Hz < PRF < {prf_max:.0f} Hz (using {p.prf:.0f} Hz)")

R, eta = p.range_axis(), p.az_axis()
sl = (slice(I - 450, I + 450), slice(J - 220, J + 220))
ext = [R[sl[1]][0] / 1e3, R[sl[1]][-1] / 1e3, eta[sl[0]][-1] * p.Vr / 1e3, eta[sl[0]][0] * p.Vr / 1e3]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(20 * np.log10(np.abs(rc[sl]) / np.abs(rc).max() + 1e-6), aspect="auto", extent=ext, vmin=-40, cmap="gray")
ax[0].set(title="Range-compressed (migrating, unfocused in azimuth)", xlabel="slant range [km]", ylabel="azimuth [km]")
ax[1].imshow(20 * np.log10(a[sl] / a.max() + 1e-6), aspect="auto", extent=ext, vmin=-40, cmap="gray")
ax[1].set(title="RDA-focused image [dB]", xlabel="slant range [km]")
th = np.linspace(-3, 3, 2000) * p.lam / p.La
ax[2].plot(np.degrees(th), 20 * np.log10(np.abs(rect_pattern(th, p.La, p.lam)) + 1e-9), label="15 m line aperture (sinc)")
ax[2].plot(np.degrees(th), 20 * np.log10(np.abs(circ_pattern(th, p.La, p.lam)) + 1e-9), label="15 m circular aperture (Airy, $2J_1(u)/u$)")
ax[2].set(ylim=(-40, 1), xlabel="angle [deg]", ylabel="one-way field [dB]", title="Fraunhofer antenna patterns")
ax[2].legend(); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig("examples/demo.png", dpi=120)
print("saved examples/demo.png")
