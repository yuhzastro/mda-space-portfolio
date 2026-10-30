"""Physical constants. Units: km, s, rad unless noted.

Sources:
  WGS-84 (NIMA TR8350.2): a, f, omega_E, GM.
  EGM-96 J2 (unnormalised).
"""
MU_EARTH = 398600.4418          # km^3/s^2, GM of Earth (WGS-84)
R_EARTH = 6378.137              # km, WGS-84 equatorial radius
F_EARTH = 1.0 / 298.257223563   # WGS-84 flattening
E2_EARTH = F_EARTH * (2.0 - F_EARTH)  # first eccentricity squared
J2 = 1.08262668e-3              # EGM-96 unnormalised J2
OMEGA_EARTH = 7.292115e-5       # rad/s, WGS-84 rotation rate
C_LIGHT = 299792.458            # km/s
SIDEREAL_DAY = 86164.0905       # s
