"""DVB-S2 MODCODs (ETSI EN 302 307-1 V1.4.1, Table 13): ideal $E_s/N_0$ for
quasi-error-free operation (PER = 1e-7, AWGN, 64800-bit FECFRAME), and
spectral efficiency in information bits per transmitted symbol (pilots off)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModCod:
    name: str
    eff: float        # bit/symbol
    esn0_db: float    # required Es/N0, ideal


MODCODS = [ModCod(*m) for m in [
    ("QPSK 1/4", 0.490243, -2.35), ("QPSK 1/3", 0.656448, -1.24), ("QPSK 2/5", 0.789412, -0.30),
    ("QPSK 1/2", 0.988858, 1.00), ("QPSK 3/5", 1.188304, 2.23), ("QPSK 2/3", 1.322253, 3.10),
    ("QPSK 3/4", 1.487473, 4.03), ("QPSK 4/5", 1.587196, 4.68), ("QPSK 5/6", 1.654663, 5.18),
    ("QPSK 8/9", 1.766451, 6.20), ("QPSK 9/10", 1.788612, 6.42),
    ("8PSK 3/5", 1.779991, 5.50), ("8PSK 2/3", 1.980636, 6.62), ("8PSK 3/4", 2.228124, 7.91),
    ("8PSK 5/6", 2.478562, 9.35), ("8PSK 8/9", 2.646012, 10.69), ("8PSK 9/10", 2.679207, 10.98),
    ("16APSK 2/3", 2.637201, 8.97), ("16APSK 3/4", 2.966728, 10.21), ("16APSK 4/5", 3.165623, 11.03),
    ("16APSK 5/6", 3.300184, 11.61), ("16APSK 8/9", 3.523143, 12.89), ("16APSK 9/10", 3.567342, 13.13),
    ("32APSK 3/4", 3.703295, 12.73), ("32APSK 4/5", 3.951571, 13.64), ("32APSK 5/6", 4.119540, 14.28),
    ("32APSK 8/9", 4.397854, 15.69), ("32APSK 9/10", 4.453027, 16.05)]]


def select_modcod(esn0_db: float, margin_db: float = 1.0, implementation_loss_db: float = 0.5):
    """Adaptive coding and modulation: the most efficient MODCOD whose
    threshold plus implementation loss plus margin is met. None if no MODCOD closes."""
    ok = [m for m in MODCODS if m.esn0_db + implementation_loss_db + margin_db <= esn0_db]
    return max(ok, key=lambda m: m.eff) if ok else None
