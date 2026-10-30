"""Satellite-to-ground QKD downlink over one pass above Montreal.

A 600 km sun-synchronous satellite sends decoy-state BB84 at 785 nm with
10 µrad divergence (as on Micius) to a 0.5 m telescope. Canada's QEYSSat
mission tests the reverse (uplink) geometry. All values are illustrative.
Channel loss uses Gaussian-beam diffraction; the key rate uses the
asymptotic decoy-state formula.
"""
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orbitkit.tle import Satellite
from orbitkit.passes import GroundStation, find_passes
from linkbudget.doppler import doppler_profile
from linkbudget.optical import downlink_efficiency
from linkbudget.qkd import decoy_key_rate

LAM, W0, D_RX, RATE = 785e-9, 785e-9 / (np.pi * 10e-6), 0.5, 100e6     # divergence lambda/(pi w0) = 10 urad
gs = GroundStation("Montreal OGS", 45.405, -73.95, 0.03, 15.0)
t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
sat = Satellite.from_elements(t0, 600.0, 97.8, raan_deg=250.0, name="QKD-sat")
p = max(find_passes(sat, gs, t0, 24.0), key=lambda x: x.max_el_deg)
ts, el, rng_km, _, _ = doppler_profile(sat, gs, p.aos, p.los, 1.0, step_s=1.0)
eta = np.array([downlink_efficiency(r * 1e3, LAM, W0, D_RX, np.radians(e)) for r, e in zip(rng_km, el)])
R = decoy_key_rate(eta)
key = np.sum(R * RATE)            # 1 s steps
print(f"pass max el {p.max_el_deg:.1f} deg, {p.duration_s:.0f} s above 15 deg")
print(f"channel loss {-10*np.log10(eta.max()):.1f} dB (zenith-most) to {-10*np.log10(eta.min()):.1f} dB (horizon)")
print(f"secret key from one pass: {key/1e3:.0f} kbit (asymptotic, no finite-size correction)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ts, -10 * np.log10(eta)); ax[0].set(xlabel="t from AOS [s]", ylabel="channel loss [dB]", title="Downlink loss over the pass")
ax[0].invert_yaxis(); ax[0].grid(alpha=.3)
ax[1].semilogy(ts, R * RATE + 1e-3); ax[1].set(xlabel="t from AOS [s]", ylabel="secret key rate [bit/s]", title="Decoy-state BB84 key rate")
ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("examples/qkd_downlink.png", dpi=120)
print("saved examples/qkd_downlink.png")
