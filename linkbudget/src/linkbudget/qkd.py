"""Quantum key distribution: BB84 with polarisation qubits (PHYS 304).

States: $|H\\rangle = (1,0)$, $|V\\rangle = (0,1)$ in the Z basis;
$|\\pm\\rangle = (|H\\rangle \\pm |V\\rangle)/\\sqrt2$ in the X basis. A measurement in basis
$\\{|b_0\\rangle, |b_1\\rangle\\}$ gives outcome $k$ with Born probability $|\\langle b_k|\\psi\\rangle|^2$.
Measuring in the wrong basis gives a random bit; this is what exposes an
eavesdropper. Intercept-resend in random bases causes a quantum bit error rate
(QBER) of 25 %.

Secret key rate: decoy-state BB84 with weak coherent pulses of mean photon
number $\\mu$ (Lo, Ma & Chen 2005; Ma et al. 2005, infinite decoys):

$R = q\\{-Q_\\mu f\\,h(E_\\mu) + Q_1[1 - h(e_1)]\\}$ per pulse, with
$Q_\\mu = Y_0 + 1 - e^{-\\eta\\mu}$, $E_\\mu Q_\\mu = e_0Y_0 + e_d(1 - e^{-\\eta\\mu})$,
$Y_1 = Y_0 + \\eta$, $Q_1 = Y_1\\mu e^{-\\mu}$, $e_1Y_1 = e_0Y_0 + e_d\\eta$,
and binary entropy $h(x) = -x\\log_2x - (1-x)\\log_2(1-x)$.
"""
from __future__ import annotations
import numpy as np

KET = {("Z", 0): np.array([1, 0], complex), ("Z", 1): np.array([0, 1], complex),
       ("X", 0): np.array([1, 1], complex) / np.sqrt(2), ("X", 1): np.array([1, -1], complex) / np.sqrt(2)}


def rotation(theta):
    """Polarisation rotation by angle theta (e.g. reference-frame misalignment)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], complex)


def measure(psi, basis, rng):
    p0 = abs(np.vdot(KET[(basis, 0)], psi)) ** 2
    return 0 if rng.random() < p0 else 1


def bb84(n, rng, eve=False, misalign=0.0):
    """Simulate n single-photon BB84 rounds. Returns (sifted key Alice, Bob)."""
    ka, kb = [], []
    U = rotation(misalign)
    for _ in range(n):
        bit, ba, bb = rng.integers(2), "ZX"[rng.integers(2)], "ZX"[rng.integers(2)]
        psi = KET[(ba, bit)]
        if eve:
            be = "ZX"[rng.integers(2)]
            psi = KET[(be, measure(psi, be, rng))]
        psi = U @ psi
        r = measure(psi, bb, rng)
        if ba == bb:
            ka.append(bit); kb.append(r)
    return np.array(ka), np.array(kb)


def h2(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def decoy_key_rate(eta, mu=0.5, Y0=1e-5, e_d=0.015, f_ec=1.16, q=0.5):
    """Asymptotic secret bits per pulse (0 if negative)."""
    e0 = 0.5
    Qmu = Y0 + 1 - np.exp(-eta * mu)
    Emu = (e0 * Y0 + e_d * (1 - np.exp(-eta * mu))) / Qmu
    Y1 = Y0 + eta
    Q1 = Y1 * mu * np.exp(-mu)
    e1 = (e0 * Y0 + e_d * eta) / Y1
    R = q * (-Qmu * f_ec * h2(Emu) + Q1 * (1 - h2(e1)))
    return np.maximum(R, 0.0)
