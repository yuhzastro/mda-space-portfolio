"""SGP4/SDP4 propagation of TLEs (Vallado et al. 2006 'Revisiting Spacetrack
Report #3', via the reference `sgp4` package), with conversion to Earth-fixed
and geodetic coordinates."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
from sgp4.api import Satrec, WGS72
from .timeutil import jd_from_datetime, gmst_iau82
from .frames import teme_to_ecef, ecef_to_geodetic


@dataclass
class State:
    t: datetime
    r_teme: np.ndarray
    v_teme: np.ndarray
    r_ecef: np.ndarray
    v_ecef: np.ndarray


class Satellite:
    """A TLE-defined satellite. SGP4 uses WGS-72 constants internally, as the
    TLE mean elements were fitted with them (Vallado 2006, sec. III)."""

    def __init__(self, line1: str, line2: str, name: str = ""):
        self.name = name
        self.sat = Satrec.twoline2rv(line1, line2, WGS72)

    @classmethod
    def from_elements(cls, epoch: datetime, alt_km: float, inc_deg: float, raan_deg: float = 0.0,
                      ecc: float = 1e-4, argp_deg: float = 0.0, mean_anom_deg: float = 0.0,
                      bstar: float = 0.0, name: str = ""):
        """Build an SGP4 satellite from mean elements (e.g. a constellation
        design point with no published TLE). Altitude is over the WGS-72
        equatorial radius, converted to Kozai mean motion to first order."""
        from sgp4.api import WGS72
        from math import sqrt, pi, radians
        mu, re = 398600.8, 6378.135              # WGS-72, as used inside SGP4
        n_rad_min = sqrt(mu / (re + alt_km) ** 3) * 60.0
        obj = cls.__new__(cls)
        obj.name = name
        obj.sat = Satrec()
        jd = jd_from_datetime(epoch)
        obj.sat.sgp4init(WGS72, "i", 99999, jd - 2433281.5, bstar, 0.0, 0.0, ecc,
                         radians(argp_deg), radians(inc_deg), radians(mean_anom_deg),
                         n_rad_min, radians(raan_deg))
        return obj

    @property
    def epoch(self) -> datetime:
        jd = self.sat.jdsatepoch + self.sat.jdsatepochF
        return datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)

    def teme(self, t: datetime):
        jd = jd_from_datetime(t)
        whole = np.floor(jd)
        err, r, v = self.sat.sgp4(whole, jd - whole)
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err} at {t}")
        return np.array(r), np.array(v)

    def state(self, t: datetime) -> State:
        r, v = self.teme(t)
        re, ve = teme_to_ecef(r, v, gmst_iau82(jd_from_datetime(t)))
        return State(t, r, v, re, ve)

    def subpoint(self, t: datetime):
        """Geodetic latitude, longitude [deg] and altitude [km]."""
        lat, lon, h = ecef_to_geodetic(self.state(t).r_ecef)
        return np.degrees(lat), np.degrees(lon), h
