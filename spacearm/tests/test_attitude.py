import numpy as np
import pytest
from scipy.integrate import solve_ivp
from spacearm.attitude import (qmul, qconj, qrot, q_to_dcm, dcm_to_q, q_angle, propagate_attitude,
                               euler_axis_stability, gravity_gradient_torque, pitch_libration_frequency,
                               pitch_period_large_amplitude, quaternion_pd_torque, inverse_laplace_residues,
                               pd_step_response, triad, davenport_q, radec_to_unit, BRIGHT_STARS, star_photon_rate)
from spacearm.proximity import cw_matrix

rng = np.random.default_rng(1)


def rand_q():
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q * np.sign(q[0])


def test_quaternion_dcm_consistency():
    for _ in range(20):
        q, v = rand_q(), rng.normal(size=3)
        assert np.allclose(qrot(q, v), q_to_dcm(q) @ v)
        assert np.allclose(dcm_to_q(q_to_dcm(q)), q)
        p = rand_q()
        assert np.allclose(q_to_dcm(qmul(p, q)), q_to_dcm(p) @ q_to_dcm(q))


# --- PHYS 216: torque-free rigid body ---------------------------------------------

def test_torque_free_conserves_energy_and_inertial_momentum():
    I = np.diag([10.0, 15.0, 20.0])
    q0, w0 = np.array([1.0, 0, 0, 0]), np.array([0.3, 0.05, -0.2])
    t, Q, W = propagate_attitude(I, q0, w0, 200.0, t_eval=np.linspace(0, 200, 400))
    E = 0.5 * np.einsum("ij,jk,ik->i", W, I, W)
    H = np.array([qrot(q, I @ w) for q, w in zip(Q, W)])
    assert np.ptp(E) / E[0] < 1e-9
    assert np.max(np.linalg.norm(H - H[0], axis=1)) / np.linalg.norm(H[0]) < 1e-8


def test_axisymmetric_body_frame_precession_rate():
    """$I_1 = I_2$: $\\omega_\\perp$ rotates in the body frame at
    $\\Omega = \\frac{I_3 - I_1}{I_1}\\omega_3$ (Taylor, *Classical Mechanics*, sec. 10.9),
    in the same sense as the spin for an oblate body ($I_3 > I_1$)."""
    I1, I3, w3 = 10.0, 16.0, 0.5
    I = np.diag([I1, I1, I3])
    t, Q, W = propagate_attitude(I, [1, 0, 0, 0], [0.02, 0.0, w3], 100.0, t_eval=np.linspace(0, 100, 2001))
    ang = np.unwrap(np.arctan2(W[:, 1], W[:, 0]))
    assert np.polyfit(t, ang, 1)[0] == pytest.approx((I3 - I1) / I1 * w3, rel=1e-8)


def test_intermediate_axis_theorem():
    I = np.array([10.0, 15.0, 20.0])
    assert euler_axis_stability(I, 0) < 0 and euler_axis_stability(I, 2) < 0
    assert euler_axis_stability(I, 1) > 0
    # nonlinear simulation: spin about the intermediate axis flips (Dzhanibekov effect)
    t, Q, W = propagate_attitude(np.diag(I), [1, 0, 0, 0], [1e-4, 1.0, 1e-4], 120.0, t_eval=np.linspace(0, 120, 3000))
    assert W[:, 1].min() < -0.9
    # growth rate matches the linear eigenvalue sqrt(coefficient) * w0
    lam = np.sqrt(euler_axis_stability(I, 1)) * 1.0
    end = int(np.argmax(np.abs(W[:, 0]) > 1e-2))            # still in the linear regime
    k = np.zeros(len(t), bool)
    k[:end] = np.abs(W[:end, 0]) > 1e-3
    rate = np.polyfit(t[k], np.log(np.abs(W[k, 0])), 1)[0]
    assert rate == pytest.approx(lam, rel=0.05)


# --- gravity gradient and the pendulum phase plane (MATH 215) -----------------------

def test_gravity_gradient_torque_vanishes_on_principal_axes():
    I = np.diag([300.0, 250.0, 100.0])
    r = np.array([7000e3, 0, 0])
    assert np.allclose(gravity_gradient_torque(3.986e14, r, [1, 0, 0, 0], I), 0.0)


def _pitch_sim(n, Ix, Iy, Iz, theta0, T):
    f = lambda t, y: [y[1], -1.5 * n * n * (Ix - Iz) / Iy * np.sin(2 * y[0])]
    return solve_ivp(f, (0, T), [theta0, 0.0], dense_output=True, rtol=1e-11, atol=1e-12, max_step=T / 2000)


@pytest.mark.parametrize("theta0_deg", [1.0, 40.0, 80.0])
def test_pitch_libration_period_elliptic_integral(theta0_deg):
    n, Ix, Iy, Iz = 2 * np.pi / 5800.0, 300.0, 250.0, 100.0
    th0 = np.radians(theta0_deg)
    Tp = pitch_period_large_amplitude(n, Ix, Iy, Iz, th0)
    sol = _pitch_sim(n, Ix, Iy, Iz, th0, 1.2 * Tp)
    # theta returns to its maximum after one period
    assert sol.sol(Tp)[0] == pytest.approx(th0, abs=1e-6)
    if theta0_deg == 1.0:
        assert Tp == pytest.approx(2 * np.pi / pitch_libration_frequency(n, Ix, Iy, Iz), rel=1e-4)


def test_cw_eigenvalues():
    """CW system matrix: eigenvalues $\\{0, 0, \\pm in, \\pm in\\}$; the zero
    eigenvalue is defective (Jordan block), which is the secular along-track drift."""
    n = 1.1e-3
    A = cw_matrix(n)
    ev = np.sort_complex(np.linalg.eigvals(A))
    assert np.allclose(np.sort(np.abs(ev.imag)), [0, 0, n, n, n, n], atol=1e-9)
    assert np.allclose(ev.real, 0, atol=1e-9)
    assert np.linalg.matrix_rank(A) == 5           # dim null(A) = 1 < algebraic multiplicity 2


# --- control: Laplace transforms by residues (MATH 215, MATH 300) -------------------

def test_inverse_laplace_by_residues_closed_form():
    t = np.linspace(0, 10, 200)
    # 1/((s+1)(s+2)) -> e^{-t} - e^{-2t}
    assert np.allclose(inverse_laplace_residues([1.0], [1.0, 3.0, 2.0], t), np.exp(-t) - np.exp(-2 * t))
    # w/(s^2+w^2) -> sin(w t): conjugate poles on the imaginary axis
    assert np.allclose(inverse_laplace_residues([2.0], [1.0, 0.0, 4.0], t), np.sin(2 * t), atol=1e-12)


def test_pd_step_response_overshoot_and_ode():
    I, Kp, Kd = 50.0, 2.0, 12.0
    wn, zeta = np.sqrt(Kp / I), Kd / (2 * np.sqrt(Kp * I))
    t = np.linspace(0, 200, 4001)
    y = pd_step_response(Kp, Kd, I, t)
    assert y.max() - 1 == pytest.approx(np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2)), rel=1e-3)
    sol = solve_ivp(lambda t, x: [x[1], (Kp * (1 - x[0]) - Kd * x[1]) / I], (0, 200), [0, 0],
                    t_eval=t, rtol=1e-11, atol=1e-12)
    assert np.allclose(sol.y[0], y, atol=1e-7)


def test_quaternion_pd_slew_converges():
    I = np.diag([120.0, 100.0, 80.0])
    q_ref = dcm_to_q(q_to_dcm(np.array([np.cos(0.8), np.sin(0.8) * 0.6, 0.0, np.sin(0.8) * 0.8])))
    t, Q, W = propagate_attitude(I, [1, 0, 0, 0], np.zeros(3), 600.0,
                                 torque=quaternion_pd_torque(q_ref, 4.0, 40.0), t_eval=[0, 600.0])
    assert np.degrees(q_angle(Q[-1], q_ref)) < 0.01
    assert np.linalg.norm(W[-1]) < 1e-5


# --- star tracker: Wahba's problem (ASTR 205) ---------------------------------------

def _stars(names):
    return [radec_to_unit(*BRIGHT_STARS[n][:2]) for n in names]


def test_wahba_exact_recovery():
    q = rand_q()
    rs = _stars(["Sirius", "Vega", "Arcturus", "Canopus"])
    bs = [q_to_dcm(q).T @ r for r in rs]
    assert q_angle(davenport_q(bs, rs), q) < 1e-10
    assert q_angle(dcm_to_q(triad(bs[0], bs[1], rs[0], rs[1])), q) < 1e-10


def test_q_method_beats_triad_with_noise():
    """With 20 arcsec noise on every star, the optimal q-method has lower mean
    error than TRIAD, which uses only two stars."""
    names = ["Sirius", "Canopus", "Arcturus", "Vega", "Capella", "Rigel", "Procyon", "Altair"]
    rs = _stars(names)
    sig = np.radians(20 / 3600)
    e_q, e_t = [], []
    for _ in range(200):
        q = rand_q()
        bs = [q_to_dcm(q).T @ r + sig * rng.normal(size=3) for r in rs]
        bs = [b / np.linalg.norm(b) for b in bs]
        e_q.append(q_angle(davenport_q(bs, rs), q))
        e_t.append(q_angle(dcm_to_q(triad(bs[0], bs[1], rs[0], rs[1])), q))
    assert np.mean(e_q) < np.mean(e_t)
    assert np.degrees(np.mean(e_q)) * 3600 < 20


def test_magnitude_scale():
    # 5 magnitudes = factor 100 in flux
    assert star_photon_rate(0.0, 1e-3) / star_photon_rate(5.0, 1e-3) == pytest.approx(100.0)
