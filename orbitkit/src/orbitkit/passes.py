"""Ground-station pass prediction (AOS / TCA / LOS) with an elevation mask.

Elevation is sampled on a coarse grid, rise/set crossings are refined by
Brent's method to 1 ms, and maximum elevation by golden-section search.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .frames import look_angles


@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_el_deg: float = 10.0

    def look(self, sat, t: datetime):
        s = sat.state(t)
        return look_angles(s.r_ecef, s.v_ecef, np.radians(self.lat_deg),
                           np.radians(self.lon_deg), self.alt_km)


@dataclass
class Pass:
    aos: datetime
    tca: datetime
    los: datetime
    max_el_deg: float
    aos_az_deg: float
    los_az_deg: float

    @property
    def duration_s(self) -> float:
        return (self.los - self.aos).total_seconds()


def find_passes(sat, gs: GroundStation, start: datetime, hours: float, step_s: float = 30.0):
    mask = np.radians(gs.min_el_deg)
    t_at = lambda x: start + timedelta(seconds=float(x))
    g = lambda x: gs.look(sat, t_at(x))[1] - mask
    ts = np.arange(0.0, hours * 3600.0 + step_s, step_s)
    gv = np.array([g(x) for x in ts])
    passes, rise = [], None
    for k in range(len(ts) - 1):
        if gv[k] < 0 <= gv[k + 1]:
            rise = brentq(g, ts[k], ts[k + 1], xtol=1e-3)
        elif gv[k] >= 0 > gv[k + 1] and rise is not None:
            sett = brentq(g, ts[k], ts[k + 1], xtol=1e-3)
            res = minimize_scalar(lambda x: -g(x), bounds=(rise, sett), method="bounded",
                                  options={"xatol": 1e-2})
            passes.append(Pass(t_at(rise), t_at(res.x), t_at(sett),
                               float(np.degrees(-res.fun + mask)),
                               float(np.degrees(gs.look(sat, t_at(rise))[0])),
                               float(np.degrees(gs.look(sat, t_at(sett))[0]))))
            rise = None
    return passes
