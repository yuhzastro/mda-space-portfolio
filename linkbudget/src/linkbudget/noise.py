"""Receiver noise: noise temperature, noise figure and cascades.

Thermal (Johnson-Nyquist) noise from a matched source at temperature $T$
delivers $N = k_BTB$ (PHYS 403, classical limit $hf \\ll k_BT$). A two-port
with noise figure $F$ has equivalent input noise temperature $T_e = T_0(F - 1)$,
where $T_0 = 290$ K.

Friis (1944) cascade:
$F = F_1 + \\frac{F_2 - 1}{G_1} + \\frac{F_3 - 1}{G_1G_2} + \\dots$, or
$T_e = T_1 + \\frac{T_2}{G_1} + \\frac{T_3}{G_1G_2} + \\dots$.
A passive lossy element at physical temperature $T_p$ has $G = 1/L$ and
$T_e = (L - 1)T_p$.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

T0 = 290.0


def db2lin(x):
    return 10 ** (np.asarray(x) / 10)


def lin2db(x):
    return 10 * np.log10(x)


@dataclass
class Stage:
    name: str
    gain_db: float
    nf_db: float | None = None     # give noise figure, or
    te_k: float | None = None      # equivalent noise temperature

    @property
    def G(self):
        return db2lin(self.gain_db)

    @property
    def Te(self):
        return self.te_k if self.te_k is not None else T0 * (db2lin(self.nf_db) - 1)


def passive_loss(name, loss_db, T_phys=T0):
    L = db2lin(loss_db)
    return Stage(name, -loss_db, te_k=(L - 1) * T_phys)


def cascade_te(stages):
    Te, G = 0.0, 1.0
    for s in stages:
        Te += s.Te / G
        G *= s.G
    return Te


def cascade_nf_db(stages):
    F, G = None, 1.0
    for s in stages:
        Fi = 1 + s.Te / T0
        F = Fi if F is None else F + (Fi - 1) / G
        G *= s.G
    return lin2db(F)


def system_temperature(T_ant, stages):
    """$T_{sys} = T_{ant} + T_{e,cascade}$, referred to the antenna terminals."""
    return T_ant + cascade_te(stages)
