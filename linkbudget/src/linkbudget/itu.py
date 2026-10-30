"""ITU-R propagation models, implemented from the Recommendations.

* ITU-R P.838-3: specific attenuation coefficients $k$, $\\alpha$ for rain,
  $\\gamma_R = k R^\\alpha$ [dB/km].
* ITU-R P.618-13 sec. 2.2.1.1: long-term rain attenuation statistics on an
  Earth-space path, exceeded for $p$ % of an average year.
* ITU-R P.839-4: rain height $h_R = h_0 + 0.36$ km. The 0 degC isotherm height
  $h_0$ comes from the digital map in P.839, so it is an input here.
* ITU-R P.837: $R_{0.01}$ (rain rate exceeded 0.01 % of the year) also comes
  from a digital map, so it is an input.
"""
from __future__ import annotations
import numpy as np

# P.838-3 Tables 1-4
_KH = dict(a=[-5.33980, -0.35351, -0.23789, -0.94158], b=[-0.10008, 1.26970, 0.86036, 0.64552],
           c=[1.13098, 0.45400, 0.15354, 0.16817], m=-0.18961, ck=0.71147)
_KV = dict(a=[-3.80595, -3.44965, -0.39902, 0.50167], b=[0.56934, -0.22911, 0.73042, 1.07319],
           c=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, ck=0.63297)
_AH = dict(a=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721], b=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
           c=[-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], m=0.67849, ck=-1.95537)
_AV = dict(a=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833], b=[2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
           c=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, ck=0.83433)


def _fit(f, t):
    lf = np.log10(f)
    return sum(a * np.exp(-((lf - b) / c) ** 2) for a, b, c in zip(t["a"], t["b"], t["c"])) + t["m"] * lf + t["ck"]


def rain_coefficients(f_ghz: float, el_deg: float = 0.0, tau_deg: float = 45.0):
    """P.838-3 eqs. (2)-(5): $k$, $\\alpha$ for path elevation $\\theta$ and
    polarisation tilt $\\tau$ (0 = horizontal, 90 = vertical, 45 = circular).
    Valid 1-1000 GHz."""
    kH, kV = 10 ** _fit(f_ghz, _KH), 10 ** _fit(f_ghz, _KV)
    aH, aV = _fit(f_ghz, _AH), _fit(f_ghz, _AV)
    c = np.cos(np.radians(el_deg)) ** 2 * np.cos(np.radians(2 * tau_deg))
    k = (kH + kV + (kH - kV) * c) / 2
    alpha = (kH * aH + kV * aV + (kH * aH - kV * aV) * c) / (2 * k)
    return float(k), float(alpha), (kH, aH, kV, aV)


def rain_attenuation(f_ghz, el_deg, lat_deg, h0_km, R001, p=0.01, hs_km=0.0, tau_deg=45.0,
                     re_km=8500.0):
    """P.618-13 sec. 2.2.1.1, steps 1-10. Returns $A_p$ [dB] exceeded for p %
    of an average year, 0.001 <= p <= 5. $R_e = 8500$ km is the effective Earth
    radius."""
    th = np.radians(el_deg)
    hR = h0_km + 0.36                                         # step 1 (P.839-4)
    if hR - hs_km <= 0 or R001 <= 0:
        return 0.0
    if el_deg >= 5:                                           # step 2
        Ls = (hR - hs_km) / np.sin(th)
    else:
        Ls = 2 * (hR - hs_km) / (np.sqrt(np.sin(th) ** 2 + 2 * (hR - hs_km) / re_km) + np.sin(th))
    LG = Ls * np.cos(th)                                      # step 3
    k, a, _ = rain_coefficients(f_ghz, el_deg, tau_deg)
    gR = k * R001 ** a                                        # step 5
    r001 = 1 / (1 + 0.78 * np.sqrt(LG * gR / f_ghz) - 0.38 * (1 - np.exp(-2 * LG)))  # step 6
    zeta = np.degrees(np.arctan2(hR - hs_km, LG * r001))      # step 7
    LR = LG * r001 / np.cos(th) if zeta > el_deg else (hR - hs_km) / np.sin(th)
    chi = 36 - abs(lat_deg) if abs(lat_deg) < 36 else 0.0
    v001 = 1 / (1 + np.sqrt(np.sin(th)) * (31 * (1 - np.exp(-el_deg / (1 + chi)))
                                           * np.sqrt(LR * gR) / f_ghz ** 2 - 0.45))
    LE = LR * v001                                            # step 8
    A001 = gR * LE                                            # step 9
    if p >= 1 or abs(lat_deg) >= 36:                          # step 10
        beta = 0.0
    elif el_deg >= 25:
        beta = -0.005 * (abs(lat_deg) - 36)
    else:
        beta = -0.005 * (abs(lat_deg) - 36) + 1.8 - 4.25 * np.sin(th)
    expo = -(0.655 + 0.033 * np.log(p) - 0.045 * np.log(A001) - beta * (1 - p) * np.sin(th))
    return float(A001 * (p / 0.01) ** expo)


def sky_noise_from_attenuation(A_db: float, T_m: float = 275.0, T_cmb: float = 2.725):
    """Brightness temperature of an absorbing atmosphere (P.618-13 sec. 3,
    radiative transfer, isothermal medium):
    $T_{sky} = T_m(1 - 10^{-A/10}) + T_{cmb}10^{-A/10}$ [K]."""
    t = 10 ** (-A_db / 10)
    return T_m * (1 - t) + T_cmb * t
