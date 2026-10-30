"""Relativistic clock corrections for navigation satellites (PHYS 200).

To first order in $1/c^2$, a clock moving at speed $v$ in potential $\\Phi$
runs at $\\frac{d\\tau}{dt} = 1 + \\frac{\\Phi}{c^2} - \\frac{v^2}{2c^2}$ relative
to coordinate time. Clocks on the rotating geoid share the potential $W_0$
(gravity + centrifugal), so the fractional rate of an orbiting clock relative
to a ground clock is

$\\frac{\\Delta f}{f} = \\frac{W_0}{c^2} - \\frac{\\mu}{rc^2} - \\frac{v^2}{2c^2}$.

On a Keplerian orbit $v^2 = \\mu(2/r - 1/a)$, which gives
$\\frac{\\Delta f}{f} = \\frac{W_0}{c^2} - \\frac{3\\mu}{2ac^2} - \\frac{2\\mu}{c^2}\\left(\\frac{1}{r} - \\frac{1}{a}\\right)$.
The last term is periodic, and integrates to the eccentricity correction
$\\Delta t_r = F e\\sqrt{a}\\sin E$ with $F = -2\\sqrt{\\mu}/c^2$ (IS-GPS-200).
"""
from __future__ import annotations
import numpy as np

C = 299_792_458.0               # m/s
MU_GPS = 3.986005e14            # m^3/s^2, value fixed by IS-GPS-200
W0 = 62_636_856.0               # m^2/s^2, geoid potential (IERS Conventions 2010; L_G = W0/c^2)


def fractional_rate_offset(a_m, mu=MU_GPS):
    """Secular fractional frequency offset of a circular-orbit clock vs. a geoid clock."""
    return W0 / C ** 2 - 1.5 * mu / (a_m * C ** 2)


def split_offsets(a_m, mu=MU_GPS):
    """(gravitational blueshift, velocity time dilation) parts, circular orbit."""
    grav = (W0 - mu / a_m) / C ** 2
    vel = -0.5 * mu / a_m / C ** 2
    return grav, vel


def eccentricity_correction(a_m, e, E, mu=MU_GPS):
    """$\\Delta t_r = -\\frac{2\\sqrt{\\mu a}}{c^2}e\\sin E$ [s]."""
    return -2 * np.sqrt(mu * a_m) / C ** 2 * e * np.sin(E)


def factory_frequency(f_nominal, a_m, mu=MU_GPS):
    """Frequency to set before launch so the clock appears nominal on the ground."""
    return f_nominal * (1 - fractional_rate_offset(a_m, mu))
