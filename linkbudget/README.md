# linkbudget

Satellite RF link budgets built from ITU-R Recommendations, with DVB-S2 adaptive coding and modulation (ACM) and Doppler over real passes.

This is the analysis that communications payload engineering starts from. MDA Space's Sainte-Anne-de-Bellevue site builds satellite payloads, antennas and electronics, including the Ka-band payloads for Telesat Lightspeed.

## What it does

| Module | Content |
|---|---|
| `itu.py` | ITU-R P.838-3 rain coefficients $k$ and $\alpha$ for any polarisation tilt $\tau$ and elevation $\theta$, giving $\gamma_R = kR^\alpha$. The full ITU-R P.618-13 §2.2.1.1 rain-attenuation procedure (steps 1–10, including the $p$ % scaling). Sky noise from attenuation, $T_{sky} = T_m(1-10^{-A/10}) + T_{CMB}10^{-A/10}$. |
| `link.py` | FSPL $20\log_{10}(4\pi d f/c)$, dish gain $\eta(\pi D/\lambda)^2$, beamwidth and pointing loss, system temperature with a lossy feed, and the $C/N_0$, $E_s/N_0$ budget with $10\log_{10}k_B = -228.60$ dBW/(K·Hz). |
| `dvbs2.py` | All 28 DVB-S2 MODCODs from ETSI EN 302 307-1 Table 13 (QEF $E_s/N_0$ and spectral efficiency), and ACM selection with margin and implementation loss. |
| `doppler.py` | $f_D = -f_0\dot\rho/c$ and $\dot f_D$ over a pass, with geometry from `orbitkit`. |

## Validation (`pytest`, 21 tests)

| Check | Reference | Result |
|---|---|---|
| P.838-3 $k_H, \alpha_H, k_V, \alpha_V$ at 10, 20 and 30 GHz | tabulated values in P.838-3 | within 0.2 % |
| P.618-13 $A_p$, 6 cases (12–30 GHz, 8–45° elevation, $p$ from 0.01 % to 1 %) | independent `itur` implementation (del Portillo et al.) | relative error $10^{-9}$ |
| Every DVB-S2 threshold above the Shannon bound $2^\eta - 1$ | Shannon | holds |
| FSPL for GEO at 12 GHz = 205.15 dB, and dish gain | textbook | within 0.01 dB |
| Doppler sign change at TCA, peak $\lvert\dot f_D\rvert$ at TCA, and $\lvert f_D\rvert < f_0 v/c$ | physics | holds |

## Run

```bash
pip install -e ../orbitkit -e .[dev]
pytest
python examples/leo_ka_downlink.py
```

The example is a 19.7 GHz user downlink from a 1015 km polar LEO satellite to a 0.6 m terminal in Montreal. It prints the clear-sky budget table, the throughput selected by ACM over the best pass of the day, and an availability table. That table uses rain fade for this site from the ITU maps: $R_{0.01} = 32.9$ mm/h and $h_0 = 3.22$ km.

![example](examples/leo_ka_downlink.png)

## Modelling notes

Gaseous and cloud attenuation is a typical zenith value at 19.7 GHz scaled by $1/\sin\theta$, rather than the full P.676 line-by-line model. Payload EIRP and terminal values are illustrative design numbers, not the specification of any flight system.
