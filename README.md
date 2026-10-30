# MDA Space portfolio: orbital mechanics, space robotics, SAR, satellite payloads

There are five Python packages. Each one builds on the physics and mathematics of my UBC BSc (astronomy) coursework, applied to what MDA Space and the wider space and defence industry actually build. Every model is derived from first principles or a published standard and tested against reference values: 144 tests in total.

| Package | Industry relevance | Highlights |
|---|---|---|
| [`orbitkit`](orbitkit) | Satellite operations, constellations (Telesat Lightspeed, AURORA), navigation | SGP4 matching Vallado's vectors to 1 mm, batch least-squares OD with $\chi^2$-consistent covariance, GPS relativistic clock corrections, eclipses, Walker coverage |
| [`spacearm`](spacearm) | Canadarm2, Canadarm3, RPO, AOCS | 7-DOF SSRMS-like kinematics and damped least squares IK, Clohessy-Wiltshire guidance, free-flyer capture, rigid-body attitude dynamics, star-tracker attitude determination |
| [`sarsim`](sarsim) | RADARSAT-2, RADARSAT Constellation Mission | Aperture diffraction, chirp pulse compression, radar equation and NESZ, Range-Doppler focusing that matches theoretical resolution to 1 % |
| [`linkbudget`](linkbudget) | Ka-band payloads (Sainte-Anne-de-Bellevue), optical and quantum links | ITU-R P.618/P.838 rain fade, Friis cascade, DVB-S2 ACM, Doppler, Gaussian-beam optics, BB84 and decoy-state QKD |
| [`spaceenv`](spaceenv) | Satellite electronics: thermal design and radiation environment | Planck and Stefan-Boltzmann from first principles, heat-equation series vs. Crank-Nicolson, Johnson noise, radiation-belt particles with a relativistic Boris pusher |

## Coursework map

| Course | Concepts used | Where |
|---|---|---|
| PHYS 200 Relativity and Quanta | time dilation, gravitational redshift, relativistic momentum | `orbitkit/relativity.py`, `spaceenv/radiation.py` |
| PHYS 203 Thermal physics | radiative balance, heat capacity | `spaceenv/thermal.py` |
| PHYS 210 Computational physics | ODE integrators (RK4, DOP853, Boris), FFT, Crank-Nicolson, Monte Carlo | throughout |
| PHYS 216 Intermediate Mechanics | central forces, rotating frames, rigid bodies, Euler's equations | `orbitkit`, `spacearm/proximity.py`, `spacearm/attitude.py` |
| PHYS 229 Experimental physics II (data analysis) | weighted least squares, covariance, $\chi^2$, Monte Carlo error | `orbitkit/od.py`, star-tracker tests |
| PHYS 301 Electricity and Magnetism | Maxwell, Poynting flux, dipole fields, Lorentz force | `sarsim/radar.py`, `sarsim/rda.py`, `spaceenv/radiation.py`, `linkbudget/link.py` |
| PHYS 304 Quantum Mechanics | state vectors, measurement, basis mismatch | `linkbudget/qkd.py` |
| PHYS 348 Frontiers in Physics and Astronomy | scientific communication | the READMEs and validation write-ups |
| PHYS 403 Statistical Mechanics | blackbody radiation, fluctuations and Johnson-Nyquist noise | `spaceenv/thermal.py`, `linkbudget/noise.py`, `sarsim/radar.py` |
| PHYS 408 Optics | Fraunhofer diffraction, Fourier optics, Gaussian beams | `sarsim/antenna.py`, `linkbudget/optical.py` |
| MATH 215 Differential Equations I | linear systems and eigenvalues, Laplace transforms, phase plane | CW stability, PD control, pitch libration |
| MATH 217 Multivariable and Vector Calculus | gradients of potentials, curl-free fields | `orbitkit/j2.py` |
| MATH 300 Complex Variables | residues, Bromwich inversion, stationary phase | `spacearm/attitude.py`, `sarsim/chirp.py` |
| MATH 316 Differential Equations II | separation of variables, Fourier series, Bessel functions, elliptic integrals | heat equation, Airy pattern, large-amplitude libration |
| ASTR 205 Stars and Stellar Populations | positions, magnitudes, stellar temperatures | star tracker, solar ephemeris, solar constant from $T_\odot$ |
| ASTR 406 High-Energy Astrophysics | charged particles in magnetic fields, radiative processes | radiation belts, Störmer cutoff |
| ASTR 200, ASTR 300 | astrophysical context; quasars define the ICRF reference frame | frames discussion |

## Run everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./orbitkit[dev]" -e "./spacearm[dev]" -e "./linkbudget[dev]" -e "./sarsim[dev]" -e "./spaceenv[dev]"
for p in orbitkit spacearm linkbudget sarsim spaceenv; do (cd $p && python -m pytest -q); done
```
