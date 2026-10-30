"""Reference frame transformations: TEME -> ECEF (PEF), geodetic, topocentric.

Polar motion (< 15 m at the surface) is neglected, so ECEF here is the
pseudo-Earth-fixed (PEF) frame of Vallado 2006.
"""
from __future__ import annotations
import numpy as np
from .constants import R_EARTH, E2_EARTH, OMEGA_EARTH


def rot3(theta: float) -> np.ndarray:
    """Passive rotation about z by angle theta: $R_3(\\theta)$."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def teme_to_ecef(r_teme, v_teme, gmst: float):
    """TEME -> PEF. $r_{PEF} = R_3(\\theta_{GMST}) r_{TEME}$ and
    $v_{PEF} = R_3(\\theta_{GMST}) v_{TEME} - \\omega_\\oplus \\times r_{PEF}$."""
    R = rot3(gmst)
    r = R @ np.asarray(r_teme, float)
    v = R @ np.asarray(v_teme, float) - np.cross([0.0, 0.0, OMEGA_EARTH], r)
    return r, v


def geodetic_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """WGS-84 geodetic (rad, rad, km) -> ECEF [km]."""
    N = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
    return np.array([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - E2_EARTH) + h) * np.sin(lat)])


def ecef_to_geodetic(r) -> tuple[float, float, float]:
    """ECEF [km] -> WGS-84 geodetic (lat rad, lon rad, h km), iterative
    solution converged to < 1e-12 rad."""
    x, y, z = r
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))
    for _ in range(20):
        N = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1.0 - E2_EARTH * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    N = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return float(lat), float(lon), float(h)


def ecef_to_enu_matrix(lat: float, lon: float) -> np.ndarray:
    """Rotation ECEF -> local East-North-Up at geodetic (lat, lon)."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def look_angles(r_sat_ecef, v_sat_ecef, lat, lon, h):
    """Azimuth (rad, from north, clockwise), elevation (rad), slant range (km)
    and range rate $\\dot\\rho$ (km/s) of a satellite from a ground station."""
    r_site = geodetic_to_ecef(lat, lon, h)
    rho = np.asarray(r_sat_ecef) - r_site
    enu = ecef_to_enu_matrix(lat, lon) @ rho
    rng = np.linalg.norm(rho)
    az = np.arctan2(enu[0], enu[1]) % (2 * np.pi)
    el = np.arcsin(enu[2] / rng)
    rdot = float(np.dot(rho, v_sat_ecef) / rng)   # station is fixed in ECEF
    return float(az), float(el), float(rng), rdot
