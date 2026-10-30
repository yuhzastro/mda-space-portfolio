"""Solar position, beta angle and eclipse geometry.

Sun position: low-precision formula of the *Astronomical Almanac* (Vallado
Algorithm 29), about 0.01 deg until 2050. It uses mean longitude, mean anomaly,
ecliptic longitude $\\lambda_{ecl} = \\lambda_M + 1.914666471\\sin M + 0.019994643\\sin 2M$,
and then rotates from ecliptic to equatorial coordinates by the obliquity
$\\varepsilon$ (spherical astronomy, ASTR 205).

Eclipse: conical shadow model with umbra and penumbra (Montenbruck & Gill,
*Satellite Orbits*, sec. 3.4.2). The visible fraction of the solar disk is the
area of overlap of two circles of angular radii $a = \\arcsin(R_\\odot/|s-r|)$ and
$b = \\arcsin(R_\\oplus/|r|)$, separated by angle $c$.
"""
from __future__ import annotations
import numpy as np
from .timeutil import jd_from_datetime

AU_KM = 149_597_870.7
R_SUN_KM = 695_700.0
R_EARTH_KM = 6378.137


def sun_position_eci(t) -> np.ndarray:
    """Geocentric Sun vector [km], mean-of-date equatorial frame (approximately TEME/TOD)."""
    T = (jd_from_datetime(t) - 2451545.0) / 36525.0
    lam_m = np.radians((280.460 + 36000.771 * T) % 360)
    M = np.radians((357.5291092 + 35999.05034 * T) % 360)
    lam = lam_m + np.radians(1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M))
    eps = np.radians(23.439291 - 0.0130042 * T)
    r = 1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2 * M)
    return r * AU_KM * np.array([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])


def sun_declination(t) -> float:
    s = sun_position_eci(t)
    return float(np.arcsin(s[2] / np.linalg.norm(s)))


def beta_angle(r, v, s) -> float:
    """Angle between the Sun direction and the orbit plane."""
    h = np.cross(r, v)
    return float(np.arcsin(np.dot(h, s) / (np.linalg.norm(h) * np.linalg.norm(s))))


def shadow_fraction(r, s) -> float:
    """Fraction of the solar disk visible from satellite position r (both km,
    geocentric): 1 sunlit, 0 umbra, between in penumbra."""
    r, s = np.asarray(r, float), np.asarray(s, float)
    d = s - r
    a = np.arcsin(R_SUN_KM / np.linalg.norm(d))
    b = np.arcsin(R_EARTH_KM / np.linalg.norm(r))
    c = np.arccos(np.clip(np.dot(-r, d) / (np.linalg.norm(r) * np.linalg.norm(d)), -1, 1))
    if c >= a + b:
        return 1.0
    if c <= b - a:
        return 0.0                                  # umbra
    if c <= a - b:
        return 1.0 - (b / a) ** 2                   # annular: Earth inside solar disk
    x = (c * c + a * a - b * b) / (2 * c)
    y = np.sqrt(max(a * a - x * x, 0.0))
    A = a * a * np.arccos(x / a) + b * b * np.arccos((c - x) / b) - c * y
    return float(1.0 - A / (np.pi * a * a))


def eclipse_fraction_cylindrical(a_km, beta):
    """Fraction of a circular orbit in a cylindrical shadow:
    $f = \\frac{1}{\\pi}\\arccos\\left(\\frac{\\sqrt{h^2 + 2R h}}{a\\cos\\beta}\\right)$
    for $|\\beta| < \\beta^* = \\arcsin(R/a)$, else 0."""
    h = a_km - R_EARTH_KM
    arg = np.sqrt(h * h + 2 * R_EARTH_KM * h) / (a_km * np.cos(beta))
    return float(np.arccos(arg) / np.pi) if arg < 1 else 0.0
