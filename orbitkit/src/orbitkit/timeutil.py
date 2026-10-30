"""Time scales. UTC is used as a proxy for UT1 (|UT1-UTC| < 0.9 s), which
introduces < 0.4 km along-track error in Earth-fixed positions at LEO."""
from __future__ import annotations
import math
from datetime import datetime, timezone


def jd_from_datetime(t: datetime) -> float:
    """Julian date (UTC) from a datetime. Naive datetimes are taken as UTC."""
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    y, m = t.year, t.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day = t.day + (t.hour + (t.minute + (t.second + t.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5


def gmst_iau82(jd_ut1: float) -> float:
    """Greenwich mean sidereal time [rad], IAU-82 model (Vallado eq. 3-47).

    $\\theta_{GMST} = 67310.54841^s + (876600^h \\cdot 3600 + 8640184.812866) T
    + 0.093104 T^2 - 6.2\\times10^{-6} T^3$, with $T$ in Julian centuries of UT1
    from J2000. This is the sidereal time SGP4's TEME frame is defined against.
    """
    t = (jd_ut1 - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t - 6.2e-6 * t ** 3)
    return math.radians((sec % 86400.0) / 240.0) % (2.0 * math.pi)
