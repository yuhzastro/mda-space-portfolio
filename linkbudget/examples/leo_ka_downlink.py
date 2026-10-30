"""Ka-band LEO user downlink over a real pass, with adaptive coding and
modulation (ACM) and rain-fade availability, for a terminal in Montreal.

Satellite: 1015 km, 98.98 deg (polar shell of a Lightspeed-class LEO design).
Payload and terminal numbers are *illustrative design values*, not the
specifications of any flight system.
"""
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orbitkit.tle import Satellite
from orbitkit.passes import GroundStation, find_passes
from linkbudget.link import LinkBudget, parabolic_gain_dbi, system_noise_temp, db
from linkbudget.itu import rain_attenuation, sky_noise_from_attenuation
from linkbudget.dvbs2 import select_modcod
from linkbudget.doppler import doppler_profile

F = 19.7e9                 # Hz, Ka downlink (17.8-20.2 GHz FSS band)
EIRP = 36.0                # dBW per beam (illustrative)
RS = 400e6                 # sym/s
D, ETA = 0.6, 0.65         # user-terminal aperture
T_LNA, L_FEED = 120.0, 0.3
A_GAS_ZENITH = 0.4         # dB, gases + cloud at 19.7 GHz, mid-latitude (typical, P.676/P.840)
LAT, LON = 45.405, -73.95
H0, R001 = 3.22, 32.9      # P.839 0 degC height [km] and P.837 R_0.01 [mm/h] at this site (from the ITU maps)

gs = GroundStation("Montreal user terminal", LAT, LON, 0.03, 20.0)
t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
sat = Satellite.from_elements(t0, 1015.0, 98.98, raan_deg=200.0, name="LEO-Ka")
p = max(find_passes(sat, gs, t0, 24.0), key=lambda x: x.max_el_deg)
ts, el, rng, fd, fdd = doppler_profile(sat, gs, p.aos, p.los, F, step_s=2.0)
G = parabolic_gain_dbi(D, F, ETA)


def budget(el_deg, rng_km, p_pct=None):
    a_gas = A_GAS_ZENITH / np.sin(np.radians(el_deg))     # cosecant law, el > 10 deg
    a_rain = rain_attenuation(F / 1e9, el_deg, LAT, H0, R001, p_pct, hs_km=0.03) if p_pct else 0.0
    t_ant = sky_noise_from_attenuation(a_gas + a_rain) + 10.0   # + ground spillover
    return LinkBudget(F, EIRP, G, system_noise_temp(t_ant, T_LNA, L_FEED), rng_km * 1e3, RS,
                      l_atm_db=a_gas + a_rain, l_point_db=0.2, l_pol_db=0.1)


print(f"Best pass: AOS {p.aos:%H:%M:%S}, LOS {p.los:%H:%M:%S} UTC, max el {p.max_el_deg:.1f} deg, "
      f"{p.duration_s:.0f} s")
k = int(np.argmax(el))
print("\nClear-sky budget at max elevation:\n" + budget(el[k], rng[k]).table())
print(f"\nMax |Doppler| {np.max(np.abs(fd))/1e3:.1f} kHz, max |Doppler rate| {np.max(np.abs(fdd)):.0f} Hz/s")

print("\nAvailability at 30 deg elevation (rain fade from ITU-R P.618-13):")
el30 = 30.0
rng30 = 6378.137 * (np.sqrt(((6378.137 + 1015) / 6378.137) ** 2 - np.cos(np.radians(el30)) ** 2)
                    - np.sin(np.radians(el30)))
for pp in (1.0, 0.5, 0.1, 0.05, 0.01):
    lb = budget(el30, rng30, pp)
    m = select_modcod(lb.es_over_n0)
    print(f"  {100-pp:6.2f} %: A_rain = {lb.l_atm_db - A_GAS_ZENITH/np.sin(np.radians(el30)):5.2f} dB, "
          f"Es/N0 = {lb.es_over_n0:5.2f} dB -> {m.name if m else 'outage':11s}"
          f" {RS*m.eff/1e6 if m else 0:7.1f} Mbit/s")

esn0 = np.array([budget(e, r).es_over_n0 for e, r in zip(el, rng)])
mods = [select_modcod(x) for x in esn0]
thr = np.array([RS * m.eff / 1e6 if m else 0 for m in mods])
print(f"\nPass-averaged ACM throughput (clear sky): {thr.mean():.0f} Mbit/s, "
      f"vs fixed worst-case MODCOD: {thr.min():.0f} Mbit/s")

fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
ax[0, 0].plot(ts, el); ax[0, 0].set(ylabel="Elevation [deg]")
ax[0, 1].plot(ts, esn0); ax[0, 1].set(ylabel="Clear-sky Es/N0 [dB]")
ax[1, 0].plot(ts, fd / 1e3); ax[1, 0].set(ylabel="Doppler [kHz]", xlabel="t from AOS [s]")
ax[1, 1].step(ts, thr, where="post"); ax[1, 1].set(ylabel="ACM throughput [Mbit/s]", xlabel="t from AOS [s]")
for a in ax.flat:
    a.grid(alpha=.3)
fig.suptitle(f"19.7 GHz LEO downlink to Montreal, best pass of 24 h (max el {p.max_el_deg:.0f}°)")
fig.tight_layout(); fig.savefig("examples/leo_ka_downlink.png", dpi=120)
print("saved examples/leo_ka_downlink.png")
