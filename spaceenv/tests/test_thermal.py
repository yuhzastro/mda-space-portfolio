import numpy as np
import pytest
from scipy.integrate import quad
from spaceenv.constants import SIGMA_SB, H, K_B, C
from spaceenv.thermal import (planck_lambda, stefan_boltzmann_from_constants, wien_constant, solar_flux,
                              planetary_equilibrium_temperature, equilibrium_temperature, band_fraction,
                              heat_series, heat_crank_nicolson, johnson_noise_psd, johnson_vrms,
                              transient_orbit, view_factor_sphere_to_earth)


def test_stefan_boltzmann_constant_from_h_k_c():
    assert stefan_boltzmann_from_constants() == pytest.approx(SIGMA_SB, rel=1e-9)


def test_bose_integral_pi4_over_15():
    """$\\int_0^\\infty x^3/(e^x-1)dx = \\Gamma(4)\\zeta(4) = 6\\cdot\\pi^4/90$."""
    val, _ = quad(lambda x: x ** 3 / np.expm1(x), 0, 200)
    assert val == pytest.approx(np.pi ** 4 / 15, rel=1e-10)


def test_planck_integrates_to_sigma_T4():
    T = 5772.0
    val, _ = quad(lambda l: np.pi * planck_lambda(l, T), 1e-8, 1e-3, limit=500, points=[5e-7])
    assert val == pytest.approx(SIGMA_SB * T ** 4, rel=1e-4)


def test_wien_constant_codata():
    # CODATA 2018: b = 2.897771955e-3 m K
    assert wien_constant() == pytest.approx(2.897771955e-3, rel=1e-9)


def test_solar_constant():
    # IAU 2015 B3 nominal total solar irradiance: 1361 W/m^2
    assert solar_flux() == pytest.approx(1361.0, rel=2e-3)


def test_earth_effective_temperature():
    # classic result: albedo 0.30 -> 255 K
    assert planetary_equilibrium_temperature(1361.0, 0.30) == pytest.approx(254.6, abs=0.3)


def test_blackbody_sphere_at_1au():
    """Sphere: absorbs on $\\pi r^2$, emits from $4\\pi r^2$: $T = (S/4\\sigma)^{1/4} = 278.3$ K."""
    T = equilibrium_temperature(1.0, 1.0, np.pi, 4 * np.pi, S=1361.0)
    assert T == pytest.approx((1361.0 / (4 * SIGMA_SB)) ** 0.25)
    assert T == pytest.approx(278.3, abs=0.1)


def test_coating_alpha_over_epsilon_controls_temperature():
    # white paint (alpha/eps ~ 0.2/0.9) much colder than gold (0.25/0.04)
    white = equilibrium_temperature(0.20, 0.90, 1.0, 1.0, S=1361.0)
    gold = equilibrium_temperature(0.25, 0.04, 1.0, 1.0, S=1361.0)
    assert white < 290 < 500 < gold
    assert gold / white == pytest.approx(((0.25 / 0.04) / (0.20 / 0.90)) ** 0.25)


def test_solar_spectrum_mostly_below_4um_earth_ir_above():
    # why alpha (solar) and epsilon (IR) are separate optical properties
    assert band_fraction(1e-7, 4e-6, 5772.0) > 0.98
    assert band_fraction(4e-6, 1e-3, 255.0) > 0.95


def test_view_factor_limits():
    assert view_factor_sphere_to_earth(1.0) == pytest.approx(0.5, abs=1e-3)
    assert view_factor_sphere_to_earth(1e12) < 1e-9


def test_heat_equation_series_vs_crank_nicolson():
    L, kappa = 0.02, 9.7e-5        # 2 cm aluminium plate, thermal diffusivity m^2/s
    x, u = heat_crank_nicolson(L, kappa, 300.0, 250.0, 280.0, t_end=0.5, nx=201, nt=4000)
    assert np.max(np.abs(u - heat_series(x, 0.5, L, kappa, 300.0, 250.0, 280.0))) < 2e-3


def test_heat_equation_decay_rate_is_first_eigenvalue():
    """Late-time decay of the transient is $e^{-\\kappa(\\pi/L)^2 t}$."""
    L, k = 1.0, 1.0
    mid = lambda t: heat_series(np.array([0.5]), t, L, k, 0.0, 0.0, 1.0)[0]
    rate = -np.log(mid(1.0) / mid(0.5)) / 0.5
    assert rate == pytest.approx(np.pi ** 2, rel=1e-6)


def test_johnson_noise_classical_and_quantum_limits():
    assert johnson_noise_psd(20e9, 290.0) == pytest.approx(K_B * 290.0, rel=2e-3)   # radio: kT
    f_opt = C / 1550e-9
    # optical (1550 nm, hf/kT = 32): suppressed by e^{-hf/kT} ~ 1e-14
    x = H * f_opt / (K_B * 290.0)
    assert johnson_noise_psd(f_opt, 290.0) == pytest.approx(H * f_opt * np.exp(-x), rel=1e-9)
    assert johnson_noise_psd(f_opt, 290.0) / (K_B * 290.0) < 1e-12
    # 1 kOhm, 290 K, 1 MHz: 4.0 uV rms (textbook)
    assert johnson_vrms(1e3, 290.0, 1e6) == pytest.approx(4.0e-6, rel=0.01)


def test_eclipse_transient_bounds():
    """Orbit transient stays between the eclipse-cooled and full-sun limits,
    and its mean is near the orbit-average steady state."""
    args = dict(alpha=0.3, eps=0.8, A_sun=0.5, A_rad=2.0)
    t, T = transient_orbit(m_c=5e3, period_s=5800.0, eclipse_frac=0.35, Q_int=40.0, **args, S=1361.0)
    T_hot = equilibrium_temperature(**args, S=1361.0, Q_int=40.0)
    T_cold = equilibrium_temperature(**args, S=1361.0, Q_int=40.0, sunlit=False)
    assert T_cold < T.min() < T.max() < T_hot
    T_avg = equilibrium_temperature(**args, S=0.65 * 1361.0, Q_int=40.0)
    assert T.mean() == pytest.approx(T_avg, abs=3.0)
