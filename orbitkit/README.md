# orbitkit

Orbit propagation, reference frames, ground-station pass prediction and LEO constellation coverage in Python. The code is validated against published reference values.

This is the flight dynamics layer that satellite operations and constellation design depend on, such as the Telesat Lightspeed constellation whose payloads MDA Space builds in Sainte-Anne-de-Bellevue, or MDA AURORA software-defined satellites.

## What it does

| Module | Content |
|---|---|
| `tle.py` | SGP4/SDP4 propagation of TLEs, using the Vallado 2006 reference implementation (`sgp4`) with WGS-72 constants. You can also build a satellite from design mean elements. |
| `timeutil.py` | Julian date, and IAU-82 GMST $\theta_{GMST}(T_{UT1})$. |
| `frames.py` | TEME → PEF, $r_{PEF} = R_3(\theta_{GMST})\,r_{TEME}$ and $v_{PEF} = R_3 v_{TEME} - \omega_\oplus \times r_{PEF}$. Also WGS-84 geodetic ↔ ECEF, ENU look angles, and range rate $\dot\rho$. |
| `kepler.py` | Classical elements ↔ state vectors, and a Kepler equation solver $M = E - e\sin E$. |
| `j2.py` | Analytic J2 secular rates $\dot\Omega = -\tfrac32 nJ_2(R_\oplus/p)^2\cos i$, $\dot\omega$ and $\dot M$, plus sun-synchronous inclination. Cowell integration (DOP853) of $\ddot r = -\mu r/\lvert r\rvert^3 + a_{J_2}$. |
| `passes.py` | AOS, TCA and LOS with an elevation mask. Crossings are refined by Brent's method to 1 ms. |
| `coverage.py` | Walker-delta $i{:}T/P/F$ shells with J2 drift, Earth central angle $\lambda = \tfrac{\pi}{2} - \varepsilon - \arcsin\!\left(\tfrac{R_\oplus}{R_\oplus+h}\cos\varepsilon\right)$, and $k$-fold coverage maps. |

## Validation (`pytest`, 17 tests)

| Check | Reference | Result |
|---|---|---|
| SGP4 satellite 00005 at $t = 0$ and 360 min | Vallado et al. 2006, AIAA 2006-6753, `tcppver.out` | within 1 mm and 1 mm/s |
| TEME → PEF | Vallado 2006 Appendix C example | within 1 mm and 0.01 mm/s |
| GMST, 1992-08-20 12:14 UT1 | Vallado, *Fundamentals* 4th ed., Ex. 3-5 | within $10^{-8}$ deg |
| Elements → $r$, $v$ | Vallado Ex. 2-6 | within 2 m |
| Numerical J2 RAAN drift vs. secular theory, ISS orbit, 5 days | first-order theory | within 0.5 %; about −5.0 deg/day |
| Sun-synchronous inclination at 800 km | standard value 98.6° | within 0.05° |
| Two-body energy conservation over 1 day | — | $\lvert\Delta E\rvert < 10^{-10}$ km²/s² |
| Range rate vs. finite difference of range | — | within 0.1 m/s |

## Run

```bash
pip install -e .[dev]
pytest
python examples/demo.py   # ISS passes over Sainte-Anne-de-Bellevue, ground track, constellation coverage map
```

![demo](examples/demo.png)

## Modelling notes

- UTC is used for UT1, which gives less than 0.4 km of along-track error. Polar motion (under 15 m) is neglected, so "ECEF" here means PEF.
- Coverage uses a spherical Earth for visibility. The two-shell, 198-satellite constellation in the demo is illustrative of a Lightspeed-class design and does not use Telesat's flight parameters.
