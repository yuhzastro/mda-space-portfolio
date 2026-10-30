"""Walker-delta constellations and ground coverage.

Coverage geometry for altitude $h$ and elevation mask $\\varepsilon$:
nadir angle $\\eta = \\arcsin\\left(\\frac{R_\\oplus}{R_\\oplus+h}\\cos\\varepsilon\\right)$,
Earth central angle $\\lambda = \\tfrac{\\pi}{2} - \\varepsilon - \\eta$
(Wertz, *Space Mission Engineering*, eq. 8-26).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .constants import R_EARTH, MU_EARTH, OMEGA_EARTH
from .kepler import coe2rv
from .j2 import secular_rates


def earth_central_angle(alt_km: float, min_el_rad: float) -> float:
    eta = np.arcsin(R_EARTH / (R_EARTH + alt_km) * np.cos(min_el_rad))
    return float(np.pi / 2 - min_el_rad - eta)


@dataclass
class WalkerShell:
    """Walker delta pattern $i: T/P/F$ (Walker 1984)."""
    alt_km: float
    inc_deg: float
    T: int
    P: int
    F: int
    raan0_deg: float = 0.0

    def elements(self):
        """(a, i, raan, u) per satellite; circular orbits, u = argument of latitude."""
        S = self.T // self.P
        a = R_EARTH + self.alt_km
        out = []
        for p in range(self.P):
            for s in range(S):
                raan = np.radians(self.raan0_deg) + 2 * np.pi * p / self.P
                u = 2 * np.pi * s / S + 2 * np.pi * self.F * p / self.T
                out.append((a, np.radians(self.inc_deg), raan, u))
        return out

    def positions_ecef(self, t_s: float, j2: bool = True):
        """Satellite ECEF positions at time t [s] after epoch (Earth at GMST=0),
        with J2 secular drift of RAAN and argument of latitude."""
        rs = []
        for a, i, raan, u in self.elements():
            n = np.sqrt(MU_EARTH / a ** 3)
            if j2:
                rd, wd, md = secular_rates(a, 0.0, i)
                raan_t, u_t = raan + rd * t_s, u + (wd + md) * t_s
            else:
                raan_t, u_t = raan, u + n * t_s
            r, _ = coe2rv(a, 0.0, i, raan_t - OMEGA_EARTH * t_s, 0.0, u_t)
            rs.append(r)
        return np.array(rs)


def coverage_fraction(shells, lats_deg, lons_deg, times_s, min_el_deg=10.0, fold=1):
    """Fraction of time each grid point sees at least `fold` satellites above
    the mask. Uses a spherical Earth for visibility (error in el < 0.2 deg)."""
    lat, lon = np.meshgrid(np.radians(lats_deg), np.radians(lons_deg), indexing="ij")
    g = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    eps = np.radians(min_el_deg)
    seen = np.zeros(lat.shape)
    for t in times_s:
        count = np.zeros(lat.shape, int)
        for sh in shells:
            lam = earth_central_angle(sh.alt_km, eps)
            r = sh.positions_ecef(t)
            u = r / np.linalg.norm(r, axis=1)[:, None]
            count += (np.tensordot(g, u, axes=([2], [1])) >= np.cos(lam)).sum(-1)
        seen += count >= fold
    return seen / len(times_s)
