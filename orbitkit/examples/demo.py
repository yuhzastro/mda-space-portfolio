"""Demo: ISS passes over MDA Sainte-Anne-de-Bellevue, ground track, and
coverage of a two-shell LEO broadband constellation.

The constellation is *illustrative* of a Lightspeed-class design (a polar
shell plus an inclined shell, 198 satellites); the parameters are not
Telesat's flight values.
"""
from datetime import timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orbitkit.tle import Satellite
from orbitkit.passes import GroundStation, find_passes
from orbitkit.coverage import WalkerShell, coverage_fraction

ISS = ("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")
gs = GroundStation("MDA Sainte-Anne-de-Bellevue", 45.405, -73.95, 0.03, 10.0)
sat = Satellite(*ISS, name="ISS")

print(f"Passes of {sat.name} over {gs.name} (mask {gs.min_el_deg} deg), 24 h from epoch {sat.epoch:%Y-%m-%d %H:%M} UTC")
print(f"{'AOS (UTC)':20s} {'TCA':9s} {'LOS':9s} {'max el':>7s} {'dur':>6s} {'AOS az':>7s} {'LOS az':>7s}")
for p in find_passes(sat, gs, sat.epoch, 24.0):
    print(f"{p.aos:%Y-%m-%d %H:%M:%S}  {p.tca:%H:%M:%S} {p.los:%H:%M:%S} {p.max_el_deg:7.2f} {p.duration_s:6.0f} "
          f"{p.aos_az_deg:7.1f} {p.los_az_deg:7.1f}")

ts = [sat.epoch + timedelta(minutes=m) for m in range(0, 3 * 92)]
lat, lon = np.array([sat.subpoint(t)[:2] for t in ts]).T
lon_plot = np.where(np.abs(np.diff(lon, prepend=lon[0])) > 180, np.nan, lon)

shells = [WalkerShell(1015.0, 98.98, 78, 6, 1), WalkerShell(1325.0, 50.88, 120, 20, 1)]
lats, lons = np.arange(-88, 89, 4.0), np.arange(-180, 180, 6.0)
cov = coverage_fraction(shells, lats, lons, np.arange(0, 2 * 3600, 120), min_el_deg=20.0, fold=2)
print(f"Area-weighted 2-fold coverage at 20 deg mask: "
      f"{np.average(cov.mean(1), weights=np.cos(np.radians(lats))):.3f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 9))
ax[0].plot(lon_plot, lat, lw=1)
ax[0].plot(gs.lon_deg, gs.lat_deg, "r*", ms=12, label=gs.name)
ax[0].set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude [deg]", ylabel="Latitude [deg]",
          title="ISS ground track, 3 orbits (SGP4)")
ax[0].legend(); ax[0].grid(alpha=.3)
im = ax[1].pcolormesh(lons, lats, cov, shading="auto", vmin=0, vmax=1)
fig.colorbar(im, ax=ax[1], label="Fraction of time with ≥2 sats above 20°")
ax[1].set(xlabel="Longitude [deg]", ylabel="Latitude [deg]", title="Two-shell LEO constellation, 198 sats")
fig.tight_layout(); fig.savefig("examples/demo.png", dpi=120)
print("saved examples/demo.png")
