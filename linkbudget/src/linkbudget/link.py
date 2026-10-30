"""RF link budget.

$\\frac{C}{N_0} = \\mathrm{EIRP} - L_{fs} - L_{atm} - L_{pt} + \\frac{G}{T} - 10\\log_{10}k_B$ [dB-Hz],
with $L_{fs} = 20\\log_{10}(4\\pi d f / c)$ and $10\\log_{10}k_B = -228.60$ dBW/(K Hz).

$\\frac{E_s}{N_0} = \\frac{C}{N_0} - 10\\log_{10} R_s$, and $\\frac{E_b}{N_0} = \\frac{E_s}{N_0} - 10\\log_{10}\\eta$
where $\\eta$ is information bits per symbol.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

C = 299_792_458.0                 # m/s
K_B = 1.380649e-23                # J/K (exact, SI 2019)
K_B_DB = 10 * np.log10(K_B)       # -228.599 dBW/(K Hz)


def db(x):
    return 10 * np.log10(x)


def fspl_db(d_m, f_hz):
    return 20 * np.log10(4 * np.pi * d_m * f_hz / C)


def parabolic_gain_dbi(D_m, f_hz, eff=0.65):
    """$G = \\eta (\\pi D / \\lambda)^2$."""
    return db(eff * (np.pi * D_m * f_hz / C) ** 2)


def hpbw_deg(D_m, f_hz, k=70.0):
    """Half-power beamwidth $\\theta_{3dB} \\approx 70\\lambda/D$ deg (uniform taper ~ 58.4, typical 65-70)."""
    return k * (C / f_hz) / D_m


def pointing_loss_db(err_deg, theta3db_deg):
    """Gaussian main-lobe approximation $L = 12(\\theta_e/\\theta_{3dB})^2$ dB."""
    return 12 * (err_deg / theta3db_deg) ** 2


def system_noise_temp(T_ant, T_lna, L_feed_db=0.0, T_phys=290.0):
    """System temperature at the LNA input, with a lossy feed of loss $L$ at
    $T_0$ in front: $T_{sys} = T_{ant}/L + T_0(1 - 1/L) + T_{LNA}$."""
    L = 10 ** (L_feed_db / 10)
    return T_ant / L + T_phys * (1 - 1 / L) + T_lna


@dataclass
class LinkBudget:
    f_hz: float
    eirp_dbw: float
    g_rx_dbi: float
    t_sys_k: float
    range_m: float
    symbol_rate: float
    l_atm_db: float = 0.0       # gases + clouds + scintillation + rain
    l_point_db: float = 0.0
    l_pol_db: float = 0.0
    l_other_db: float = 0.0

    @property
    def g_over_t(self):
        return self.g_rx_dbi - db(self.t_sys_k)

    @property
    def fspl(self):
        return fspl_db(self.range_m, self.f_hz)

    @property
    def c_over_n0(self):
        return (self.eirp_dbw - self.fspl - self.l_atm_db - self.l_point_db - self.l_pol_db
                - self.l_other_db + self.g_over_t - K_B_DB)

    @property
    def es_over_n0(self):
        return self.c_over_n0 - db(self.symbol_rate)

    def table(self):
        rows = [("Frequency [GHz]", self.f_hz / 1e9), ("EIRP [dBW]", self.eirp_dbw),
                ("Slant range [km]", self.range_m / 1e3), ("Free-space loss [dB]", -self.fspl),
                ("Atmospheric loss [dB]", -self.l_atm_db), ("Pointing loss [dB]", -self.l_point_db),
                ("Polarisation loss [dB]", -self.l_pol_db), ("Other losses [dB]", -self.l_other_db),
                ("Rx G/T [dB/K]", self.g_over_t), ("-10 log k [dBW/K/Hz]", -K_B_DB),
                ("C/N0 [dB-Hz]", self.c_over_n0), ("Symbol rate [Msym/s]", self.symbol_rate / 1e6),
                ("Es/N0 [dB]", self.es_over_n0)]
        return "\n".join(f"{k:26s}{v:12.2f}" for k, v in rows)
