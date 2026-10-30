"""Demo: (1) Sun vs Earth blackbody spectra and why coatings have separate
alpha and epsilon; (2) orbit temperature of a 1015 km LEO electronics box with
eclipses; (3) a 10 MeV proton trapped at L = 3, integrated with the Lorentz
force, with gyration, bounce and westward drift.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spaceenv.constants import Q_E, M_P, R_EARTH
from spaceenv.thermal import planck_lambda, transient_orbit, equilibrium_temperature, view_factor_sphere_to_earth
from spaceenv.radiation import speed, boris_push, bounce_period, drift_period, mirror_latitude, loss_cone

# (2) box: 20 kg aluminium (c = 900 J/kg/K), 0.25 m^2 sun-facing, 0.8 m^2 radiating, 120 W dissipation
h = 1015e3
a_orb = R_EARTH + h
period = 2 * np.pi * np.sqrt(a_orb ** 3 / 3.986004418e14)
ecl = np.arcsin(R_EARTH / a_orb) / np.pi              # beta = 0 cylindrical-shadow eclipse fraction
F = view_factor_sphere_to_earth(h)
args = dict(alpha=0.25, eps=0.85, A_sun=0.25, A_rad=0.8, Q_int=120.0, F_earth=F)
t, T = transient_orbit(m_c=20 * 900.0, period_s=period, eclipse_frac=ecl, **args)
print(f"orbit period {period/60:.1f} min, eclipse {ecl*100:.1f} % (beta = 0), Earth view factor {F:.3f}")
print(f"box temperature over an orbit: {T.min()-273.15:.1f} to {T.max()-273.15:.1f} degC "
      f"(steady-state limits {equilibrium_temperature(**args, sunlit=False)-273.15:.1f} / "
      f"{equilibrium_temperature(**args)-273.15:.1f} degC)")

# (3) radiation belt proton
L, W, al = 3.0, 10e6 * Q_E, np.radians(45)
v = speed(W, M_P)
ts, xs, vs = boris_push([L * R_EARTH, 0, 0], v * np.array([0, np.sin(al), np.cos(al)]), Q_E, M_P, 60.0, 5e-4)
print(f"10 MeV proton at L = 3: bounce {bounce_period(L, W, M_P, al):.2f} s, drift {drift_period(L, W, M_P, al):.1f} s, "
      f"mirror latitude {np.degrees(mirror_latitude(al)):.1f} deg, loss cone {np.degrees(loss_cone(L)):.1f} deg")

fig = plt.figure(figsize=(16, 4.8))
ax1 = fig.add_subplot(1, 3, 1)
lam = np.geomspace(0.1e-6, 100e-6, 800)
ax1.loglog(lam * 1e6, np.pi * planck_lambda(lam, 5772) * (6.957e8 / 1.496e11) ** 2 * 1e-6, label="Sun at 1 AU (5772 K)")
ax1.loglog(lam * 1e6, np.pi * planck_lambda(lam, 255) * 1e-6, label="Earth (255 K)")
ax1.set(ylim=(1e-3, 3e3), xlabel="wavelength [µm]", ylabel="W m$^{-2}$ µm$^{-1}$", title="Planck spectra: solar vs IR band")
ax1.legend(); ax1.grid(alpha=.3, which="both")
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot((t - t[0]) / 60, T - 273.15)
ax2.axvspan((1 - ecl) * period / 60, period / 60, color="k", alpha=.15, label="eclipse")
ax2.set(xlabel="time in orbit [min]", ylabel="temperature [°C]", title="Electronics box, 1015 km LEO")
ax2.legend(); ax2.grid(alpha=.3)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot(*(xs.T / R_EARTH), lw=0.3)
u, w = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
ax3.plot_surface(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), color="tab:blue", alpha=.3)
ax3.set(title="10 MeV proton, L = 3, 60 s (Boris pusher)", xlabel="x [R$_E$]", ylabel="y [R$_E$]", zlabel="z [R$_E$]")
fig.tight_layout(); fig.savefig("examples/demo.png", dpi=120)
print("saved examples/demo.png")
