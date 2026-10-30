"""Radiation and heat transfer for spacecraft thermal design.

Planck's law (PHYS 403, photon gas): spectral radiance
$B_\\lambda(T) = \\frac{2hc^2}{\\lambda^5}\\frac{1}{e^{hc/\\lambda k_BT} - 1}$.
Integrating over wavelength and the hemisphere gives Stefan-Boltzmann,
$\\int_0^\\infty \\pi B_\\lambda d\\lambda = \\sigma T^4$ with
$\\sigma = \\frac{2\\pi^5 k_B^4}{15 h^3 c^2}$, from $\\int_0^\\infty \\frac{x^3}{e^x - 1}dx = \\Gamma(4)\\zeta(4) = \\frac{\\pi^4}{15}$.

Wien's displacement law: $\\lambda_{max}T = \\frac{hc}{x^* k_B}$, where $x^*$ solves
$x = 5(1 - e^{-x})$, so $x^* = 5 + W_0(-5e^{-5})$ (Lambert W).
"""
from __future__ import annotations
import numpy as np
from scipy.special import lambertw
from .constants import H, K_B, C, SIGMA_SB, AU, R_SUN, T_SUN


def planck_lambda(lam, T):
    """Spectral radiance [W m^-2 sr^-1 m^-1]."""
    x = H * C / (lam * K_B * T)
    return 2 * H * C ** 2 / lam ** 5 / np.expm1(x)


def stefan_boltzmann_from_constants():
    return 2 * np.pi ** 5 * K_B ** 4 / (15 * H ** 3 * C ** 2)


def wien_constant():
    x = 5 + lambertw(-5 * np.exp(-5)).real
    return H * C / (x * K_B)


def solar_flux(r_m=AU):
    """Irradiance from a blackbody Sun: $S = \\sigma T_\\odot^4 (R_\\odot / r)^2$."""
    return SIGMA_SB * T_SUN ** 4 * (R_SUN / r_m) ** 2


def band_fraction(lam1, lam2, T, n=20000):
    """Fraction of blackbody exitance between two wavelengths."""
    lam = np.geomspace(lam1, lam2, n)
    return np.trapezoid(np.pi * planck_lambda(lam, T), lam) / (SIGMA_SB * T ** 4)


def planetary_equilibrium_temperature(S, albedo):
    """Fast-rotating blackbody planet: $T = \\left[\\frac{S(1-a)}{4\\sigma}\\right]^{1/4}$."""
    return (S * (1 - albedo) / (4 * SIGMA_SB)) ** 0.25


def view_factor_sphere_to_earth(alt_m, R=6.3712e6):
    """View factor from a small sphere to Earth: $F = \\tfrac{1}{2}\\left(1 - \\sqrt{1 - (R/r)^2}\\right)$."""
    return 0.5 * (1 - np.sqrt(1 - (R / (R + alt_m)) ** 2))


def equilibrium_temperature(alpha, eps, A_sun, A_rad, S=None, Q_int=0.0, albedo=0.30, F_earth=0.0,
                            q_ir=237.0, sunlit=True):
    """Steady radiative balance of an isothermal body:
    $\\alpha S A_{sun} + \\alpha a S F A_{rad} + \\varepsilon q_{IR} F A_{rad} + Q_{int} = \\varepsilon\\sigma A_{rad} T^4$.
    $q_{IR} \\approx 237$ W/m$^2$ is Earth's mean outgoing longwave radiation; the
    albedo term uses the subsolar maximum and is conservative (hot case)."""
    S = solar_flux() if S is None else S
    q = (alpha * S * A_sun + alpha * albedo * S * F_earth * A_rad) if sunlit else 0.0
    q += eps * q_ir * F_earth * A_rad + Q_int
    return (q / (eps * SIGMA_SB * A_rad)) ** 0.25


def transient_orbit(m_c, alpha, eps, A_sun, A_rad, period_s, eclipse_frac, Q_int=0.0, F_earth=0.0,
                    T0=280.0, n_orbits=20, steps_per_orbit=2000, S=None, albedo=0.30, q_ir=237.0):
    """Lumped-capacitance transient $mc\\,\\dot T = Q_{in}(t) - \\varepsilon\\sigma A T^4$
    over repeated orbits with an eclipse, RK4. Returns (t, T) for the last orbit."""
    S = solar_flux() if S is None else S
    dt = period_s / steps_per_orbit

    def qin(t):
        lit = (t % period_s) < (1 - eclipse_frac) * period_s
        q = eps * q_ir * F_earth * A_rad + Q_int
        if lit:
            q += alpha * S * A_sun + alpha * albedo * S * F_earth * A_rad
        return q

    f = lambda t, T: (qin(t) - eps * SIGMA_SB * A_rad * T ** 4) / m_c
    T, t = T0, 0.0
    ts, Ts = [], []
    for k in range(n_orbits * steps_per_orbit):
        k1 = f(t, T); k2 = f(t + dt / 2, T + dt / 2 * k1)
        k3 = f(t + dt / 2, T + dt / 2 * k2); k4 = f(t + dt, T + dt * k3)
        T += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if k >= (n_orbits - 1) * steps_per_orbit:
            ts.append(t); Ts.append(T)
    return np.array(ts), np.array(Ts)


# --- conduction: 1-D heat equation ------------------------------------------

def heat_series(x, t, L, kappa, T_left, T_right, T_init, n_terms=400):
    """Exact solution of $u_t = \\kappa u_{xx}$ on $[0, L]$, $u(0)=T_l$, $u(L)=T_r$,
    $u(x,0) = T_{init}$ (uniform), by separation of variables (MATH 316):

    $u = T_l + (T_r - T_l)\\frac{x}{L} + \\sum_{n\\ge1} b_n \\sin\\frac{n\\pi x}{L}e^{-\\kappa (n\\pi/L)^2 t}$,
    $b_n = \\frac{2}{L}\\int_0^L \\left[T_{init} - T_l - (T_r - T_l)\\tfrac{x}{L}\\right]\\sin\\frac{n\\pi x}{L}dx
    = \\frac{2}{n\\pi}\\left[(T_{init} - T_l)(1 - (-1)^n) + (T_r - T_l)(-1)^n\\right]$.
    """
    x = np.asarray(x, float)
    u = T_left + (T_right - T_left) * x / L
    for n in range(1, n_terms + 1):
        bn = 2 / (n * np.pi) * ((T_init - T_left) * (1 - (-1) ** n) + (T_right - T_left) * (-1) ** n)
        u = u + bn * np.sin(n * np.pi * x / L) * np.exp(-kappa * (n * np.pi / L) ** 2 * t)
    return u


def heat_crank_nicolson(L, kappa, T_left, T_right, T_init, t_end, nx=201, nt=2000):
    """Second-order implicit finite differences (Crank-Nicolson), unconditionally stable."""
    from scipy.linalg import solve_banded
    x = np.linspace(0, L, nx)
    dx, dt = x[1] - x[0], t_end / nt
    r = kappa * dt / dx ** 2
    u = np.full(nx, float(T_init))
    u[0], u[-1] = T_left, T_right
    m = nx - 2
    ab = np.zeros((3, m))
    ab[0, 1:] = -r / 2; ab[1, :] = 1 + r; ab[2, :-1] = -r / 2
    for _ in range(nt):
        rhs = r / 2 * u[:-2] + (1 - r) * u[1:-1] + r / 2 * u[2:]
        rhs[0] += r / 2 * T_left
        rhs[-1] += r / 2 * T_right
        u[1:-1] = solve_banded((1, 1), ab, rhs)
    return x, u


# --- thermal noise ----------------------------------------------------------

def johnson_noise_psd(f, T):
    """Available noise power spectral density [W/Hz] of a resistor (Nyquist 1928)
    with the quantum correction: $\\frac{hf}{e^{hf/k_BT} - 1}$. This tends to $k_BT$
    when $hf \\ll k_BT$ (radio), and is exponentially small at optical
    frequencies, where photon shot noise dominates instead."""
    x = H * np.asarray(f, float) / (K_B * T)
    return K_B * T * np.where(x < 1e-8, 1 - x / 2, x / np.expm1(x))


def johnson_vrms(R, T, B):
    """Open-circuit RMS noise voltage $\\sqrt{4k_BTRB}$."""
    return np.sqrt(4 * K_B * T * R * B)
