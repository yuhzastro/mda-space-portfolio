"""Stripmap SAR raw-data simulation and Range-Doppler Algorithm (RDA) focusing.

Geometry: straight-line, zero-squint, effective velocity $V_r$. A point
target at closest-approach range $R_0$ and azimuth time $\\eta_c$ has range history
$R(\\eta) = \\sqrt{R_0^2 + V_r^2(\\eta-\\eta_c)^2}$. Demodulated baseband echo:

$s(\\tau,\\eta) = w_a(\\eta)\\,\\mathrm{rect}\\!\\left(\\frac{\\tau - 2R/c}{T_p}\\right)
e^{-i4\\pi f_0 R(\\eta)/c}\\,e^{i\\pi K(\\tau - 2R/c)^2}.$

RDA (Cumming & Wong, *Digital Processing of SAR Data*, ch. 6):
1. range compression (matched filter);
2. azimuth FFT to the range-Doppler domain, where all targets at the same
   $R_0$ collapse onto one trajectory $R(f_\\eta) = R_0/D(f_\\eta)$,
   $D = \\sqrt{1 - \\left(\\frac{\\lambda f_\\eta}{2V_r}\\right)^2}$;
3. range cell migration correction (shift by $R_0(1/D - 1)$);
4. azimuth matched filter $H_{az} = \\exp(i4\\pi R_0 (D-1) f_0/c)$, inverse FFT.
   The $-1$ keeps the phase of each target at its natural value
   $-4\\pi f_0 R_0/c$ across its whole impulse response. Using $D$ alone would
   impose a phase ramp of $4\\pi f_0/c$ per metre of range, which shifts the
   range spectrum by $2f_0\\Delta r/c$ cycles per sample (mod 1) and wraps it.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .chirp import chirp, matched_filter

C = 299_792_458.0


@dataclass
class SARParams:
    f0: float = 5.405e9         # Hz, C-band (RADARSAT-2 / RCM centre frequency)
    B: float = 30e6             # Hz, chirp bandwidth
    Tp: float = 20e-6           # s, pulse length
    fs: float = 36e6            # Hz, range sampling (1.2x oversampled)
    prf: float = 1700.0         # Hz
    Vr: float = 7100.0          # m/s, effective (rectilinear) velocity
    La: float = 15.0            # m, antenna length in azimuth
    R_ref: float = 850e3        # m, scene-centre slant range
    n_rg: int = 2048
    n_az: int = 2048

    @property
    def lam(self):
        return C / self.f0

    @property
    def K(self):
        return self.B / self.Tp

    def range_axis(self):
        """Slant range [m] of each range sample; the window opens at R_ref - span/2."""
        dr = C / (2 * self.fs)
        return self.R_ref + (np.arange(self.n_rg) - self.n_rg // 2) * dr

    def az_axis(self):
        return (np.arange(self.n_az) - self.n_az // 2) / self.prf


def simulate_raw(p: SARParams, targets, noise_std=0.0, rng=None):
    """targets: iterable of (R0 [m], eta_c [s], complex amplitude)."""
    tau = 2 * p.range_axis() / C
    eta = p.az_axis()
    raw = np.zeros((p.n_az, p.n_rg), complex)
    for R0, eta_c, amp in targets:
        R = np.sqrt(R0 ** 2 + (p.Vr * (eta - eta_c)) ** 2)
        theta = np.arctan2(p.Vr * (eta - eta_c), R0)
        wa = np.sinc(p.La * np.sin(theta) / p.lam) ** 2        # two-way pattern
        wa[np.abs(np.sin(theta)) > p.lam / p.La] = 0.0          # main lobe only
        td = tau[None, :] - 2 * R[:, None] / C
        env = np.abs(td) <= p.Tp / 2
        raw += amp * wa[:, None] * env * np.exp(-4j * np.pi * p.f0 * R[:, None] / C
                                                + 1j * np.pi * p.K * td ** 2)
    if noise_std:
        rng = rng or np.random.default_rng()
        raw += noise_std / np.sqrt(2) * (rng.normal(size=raw.shape) + 1j * rng.normal(size=raw.shape))
    return raw


def range_compress(p: SARParams, raw):
    _, ref = chirp(p.Tp, p.B, p.fs)
    n = int(2 ** np.ceil(np.log2(p.n_rg + len(ref))))
    REF = np.conj(np.fft.fft(ref, n))
    out = np.fft.ifft(np.fft.fft(raw, n, axis=1) * REF[None, :], axis=1)
    # ref is centred, so the peak for delay td sits len(ref)/2 samples early: re-align
    return np.roll(out, len(ref) // 2, axis=1)[:, :p.n_rg]


def rda(p: SARParams, raw):
    """Focus raw data. Returns the complex image on (az_axis, range_axis)."""
    rc = range_compress(p, raw)
    S = np.fft.fft(rc, axis=0)                         # range-Doppler domain
    f_eta = np.fft.fftfreq(p.n_az, 1 / p.prf)
    D = np.sqrt(1 - (p.lam * f_eta / (2 * p.Vr)) ** 2)
    R0 = p.range_axis()
    # RCMC: shift each Doppler row left by dR = R_ref (1/D - 1) using an FFT phase ramp
    dR = p.R_ref * (1 / D - 1)
    fr = np.fft.fftfreq(p.n_rg, 1 / p.fs)
    S = np.fft.ifft(np.fft.fft(S, axis=1) * np.exp(1j * 2 * np.pi * fr[None, :] * (2 * dR[:, None] / C)), axis=1)
    # azimuth matched filter, range dependent
    Haz = np.exp(4j * np.pi * p.f0 * R0[None, :] * (D[:, None] - 1.0) / C)
    # H_az removes only the hyperbolic phase, so each target keeps its own
    # linear phase and is focused at its own azimuth index
    return np.fft.ifft(S * Haz, axis=0)
