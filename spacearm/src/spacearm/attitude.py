"""Rigid-body attitude dynamics, determination and control.

Quaternions are scalar-first Hamilton quaternions $q = (q_0, \\mathbf q_v)$,
mapping body to inertial: $v_I = q \\otimes v_B \\otimes q^*$.

Dynamics (PHYS 216, Euler's equations in the rotating body frame):
$I\\dot\\omega + \\omega\\times I\\omega = \\tau$, with kinematics $\\dot q = \\tfrac12 q\\otimes(0, \\omega)$.

Gravity-gradient torque on a body at orbit radius $r$ with nadir unit vector
$\\hat r_B$ in body axes: $\\tau_{gg} = \\frac{3\\mu}{r^3}\\,\\hat r_B\\times I\\hat r_B$.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


# --- quaternion algebra -------------------------------------------------------

def qmul(p, q):
    p0, pv, q0, qv = p[0], np.asarray(p[1:]), q[0], np.asarray(q[1:])
    return np.concatenate([[p0 * q0 - pv @ qv], p0 * qv + q0 * pv + np.cross(pv, qv)])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    """Rotate vector v from body to inertial."""
    return qmul(qmul(q, np.concatenate([[0.0], v])), qconj(q))[1:]


def q_to_dcm(q):
    """Direction cosine matrix C with v_I = C v_B."""
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]])


def dcm_to_q(C):
    """Shepperd's method (numerically robust)."""
    tr = np.trace(C)
    cands = [1 + tr, 1 + 2 * C[0, 0] - tr, 1 + 2 * C[1, 1] - tr, 1 + 2 * C[2, 2] - tr]
    k = int(np.argmax(cands))
    s = np.sqrt(cands[k]) * 2
    if k == 0:
        q = [s / 4, (C[2, 1] - C[1, 2]) / s, (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s]
    elif k == 1:
        q = [(C[2, 1] - C[1, 2]) / s, s / 4, (C[0, 1] + C[1, 0]) / s, (C[0, 2] + C[2, 0]) / s]
    elif k == 2:
        q = [(C[0, 2] - C[2, 0]) / s, (C[0, 1] + C[1, 0]) / s, s / 4, (C[1, 2] + C[2, 1]) / s]
    else:
        q = [(C[1, 0] - C[0, 1]) / s, (C[0, 2] + C[2, 0]) / s, (C[1, 2] + C[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def q_angle(q1, q2):
    """Rotation angle between two attitudes [rad]."""
    d = abs(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2))
    return 2 * np.arccos(min(d, 1.0))


# --- dynamics -----------------------------------------------------------------

def rigid_body_rhs(I, torque=None):
    Iinv = np.linalg.inv(I)

    def f(t, y):
        q, w = y[:4], y[4:]
        tau = np.zeros(3) if torque is None else torque(t, q, w)
        qd = 0.5 * qmul(q, np.concatenate([[0.0], w]))
        wd = Iinv @ (tau - np.cross(w, I @ w))
        return np.concatenate([qd, wd])
    return f


def propagate_attitude(I, q0, w0, t_end, torque=None, t_eval=None, rtol=1e-11, atol=1e-12):
    sol = solve_ivp(rigid_body_rhs(I, torque), (0, t_end), np.concatenate([q0, w0]),
                    method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    return sol.t, sol.y[:4].T, sol.y[4:].T


def euler_axis_stability(I_diag, axis):
    """Linearised Euler equations about steady spin $\\omega_0$ on principal axis k:
    perturbations obey $\\ddot\\epsilon = \\frac{(I_k - I_i)(I_j - I_k)}{I_iI_j}\\omega_0^2\\,\\epsilon$
    (MATH 215: sign of the eigenvalue decides). Returns the coefficient per
    $\\omega_0^2$: negative means oscillatory (stable), positive means
    exponential growth (the intermediate axis)."""
    k = axis
    i, j = [a for a in range(3) if a != k]
    Ii, Ij, Ik = I_diag[i], I_diag[j], I_diag[k]
    return (Ik - Ii) * (Ij - Ik) / (Ii * Ij)


def gravity_gradient_torque(mu, r_vec_I, q, I):
    """$\\tau = \\frac{3\\mu}{r^3}\\hat r_B\\times I\\hat r_B$, with r_vec in the inertial frame."""
    r = np.linalg.norm(r_vec_I)
    rb = q_to_dcm(q).T @ (np.asarray(r_vec_I) / r)
    return 3 * mu / r ** 3 * np.cross(rb, I @ rb)


def pitch_libration_frequency(n, Ix, Iy, Iz):
    """Small-amplitude pitch libration of a gravity-gradient stabilised satellite
    (pitch about orbit normal y, x along-track, z nadir):
    $\\omega_\\theta = n\\sqrt{3(I_x - I_z)/I_y}$."""
    return n * np.sqrt(3 * (Ix - Iz) / Iy)


def pitch_period_large_amplitude(n, Ix, Iy, Iz, theta_max):
    """Exact period of $\\ddot\\theta = -\\tfrac{3}{2}n^2\\frac{I_x - I_z}{I_y}\\sin 2\\theta$.
    With $\\phi = 2\\theta$ this is a pendulum, so $T = \\frac{4K(k)}{\\omega_\\theta}$,
    $k = \\sin\\theta_{max}$, where $K$ is the complete elliptic integral of the first kind."""
    from scipy.special import ellipk
    return 4 * ellipk(np.sin(theta_max) ** 2) / pitch_libration_frequency(n, Ix, Iy, Iz)


# --- control: PD law, closed-loop response by residues ---------------------------

def quaternion_pd_torque(q_ref, Kp, Kd):
    """Wie, Weiss & Arapostathis (1989): $\\tau = -K_p\\,\\mathbf q_{e,v}\\,\\mathrm{sgn}(q_{e,0}) - K_d\\,\\omega$,
    with $q_e = q_{ref}^*\\otimes q$. Globally stabilising for a rest-to-rest slew."""
    def tau(t, q, w):
        qe = qmul(qconj(q_ref), q)
        return -Kp * qe[1:] * np.sign(qe[0] if qe[0] != 0 else 1.0) - Kd * w
    return tau


def inverse_laplace_residues(num, den, t):
    """$f(t) = \\mathcal L^{-1}\\{N(s)/D(s)\\}$ for a strictly proper rational
    function with simple poles, by the residue theorem (Bromwich contour closed
    to the left, MATH 300): $f(t) = \\sum_k \\frac{N(s_k)}{D'(s_k)}e^{s_kt}$."""
    num, den = np.poly1d(num), np.poly1d(den)
    dden = den.deriv()
    t = np.asarray(t, float)
    out = np.zeros_like(t, dtype=complex)
    for s in den.roots:
        out += num(s) / dden(s) * np.exp(s * t)
    return out.real


def pd_step_response(Kp, Kd, I, t):
    """Single-axis $I\\ddot\\theta = K_p(\\theta_c - \\theta) - K_d\\dot\\theta$: unit
    step response $\\Theta(s) = \\frac{\\omega_n^2}{s(s^2 + 2\\zeta\\omega_n s + \\omega_n^2)}$,
    $\\omega_n = \\sqrt{K_p/I}$, $\\zeta = K_d/(2\\sqrt{K_pI})$."""
    wn2 = Kp / I
    return inverse_laplace_residues([wn2], [1.0, Kd / I, wn2, 0.0], t)


# --- determination: Wahba's problem -------------------------------------------

def triad(b1, b2, r1, r2):
    """TRIAD (Black 1964): C with r = C b from two vector pairs (body b, reference r).
    The first pair is trusted exactly."""
    def frame(a, b):
        t1 = a / np.linalg.norm(a)
        t2 = np.cross(a, b); t2 /= np.linalg.norm(t2)
        return np.column_stack([t1, t2, np.cross(t1, t2)])
    return frame(r1, r2) @ frame(b1, b2).T


def davenport_q(bs, rs, w=None):
    """Optimal solution of Wahba's problem, $\\min_C \\tfrac12\\sum w_i|r_i - Cb_i|^2$,
    by Davenport's q-method: q is the eigenvector of the 4x4 matrix K with the
    largest eigenvalue. Returns q (body to reference)."""
    bs, rs = np.atleast_2d(bs), np.atleast_2d(rs)
    w = np.ones(len(bs)) if w is None else np.asarray(w)
    B = sum(wi * np.outer(ri, bi) for wi, ri, bi in zip(w, rs, bs))
    S = B + B.T
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    K = np.zeros((4, 4))
    K[0, 0] = np.trace(B)
    K[0, 1:] = K[1:, 0] = z
    K[1:, 1:] = S - np.trace(B) * np.eye(3)
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, -1]
    # eigen-solution gives the reference-to-body sense for this K convention; return body-to-reference
    q = qconj(q)
    return q if q[0] >= 0 else -q


def radec_to_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def star_photon_rate(V_mag, aperture_m2, band_m=88e-9, qe=0.7, throughput=0.8):
    """Detected photo-electrons per second from a star of visual magnitude V
    (ASTR 205: $m = -2.5\\log_{10}F + \\mathrm{const}$). V-band zero point is
    about $10^3$ photons s$^{-1}$ cm$^{-2}$ Å$^{-1}$ (Bessell 1998)."""
    zp = 1e3 * 1e4 * 1e10          # photons s^-1 m^-2 m^-1
    return zp * band_m * 10 ** (-0.4 * V_mag) * aperture_m2 * qe * throughput


# J2000 positions (RA, Dec in deg) and V magnitudes of bright navigation stars
BRIGHT_STARS = {
    "Sirius": (101.2872, -16.7161, -1.46), "Canopus": (95.9880, -52.6957, -0.74),
    "Arcturus": (213.9153, 19.1824, -0.05), "Vega": (279.2347, 38.7837, 0.03),
    "Capella": (79.1723, 45.9980, 0.08), "Rigel": (78.6345, -8.2016, 0.13),
    "Procyon": (114.8255, 5.2250, 0.34), "Betelgeuse": (88.7929, 7.4071, 0.50),
    "Altair": (297.6958, 8.8683, 0.76), "Aldebaran": (68.9802, 16.5093, 0.86),
    "Spica": (201.2983, -11.1613, 0.97), "Antares": (247.3519, -26.4320, 1.09),
    "Deneb": (310.3580, 45.2803, 1.25), "Polaris": (37.9546, 89.2641, 1.98),
}
