"""Rotation utilities."""
from __future__ import annotations
import numpy as np


def skew(w) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def log_so3(R) -> np.ndarray:
    """Rotation vector $\\phi\\hat u$ with $R = \\exp([\\phi \\hat u]_\\times)$, $0 \\le \\phi \\le \\pi$."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    phi = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if phi < 1e-9:
        return 0.5 * v
    if np.pi - phi < 1e-6:
        # near pi: axis from the symmetric part, R = 2uu^T - I
        u = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(u))
        u[[j for j in range(3) if j != k]] = (R[k, [j for j in range(3) if j != k]]
                                             + R[[j for j in range(3) if j != k], k]) / (4 * u[k])
        return phi * u / np.linalg.norm(u)
    return phi / (2.0 * np.sin(phi)) * v


def exp_so3(w) -> np.ndarray:
    """Rodrigues' formula $R = I + \\frac{\\sin\\phi}{\\phi}[w]_\\times + \\frac{1-\\cos\\phi}{\\phi^2}[w]_\\times^2$."""
    phi = np.linalg.norm(w)
    K = skew(w)
    if phi < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(phi) / phi * K + (1 - np.cos(phi)) / phi ** 2 * K @ K


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
