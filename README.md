# MDA Space portfolio: orbital mechanics, space robotics, satellite payloads

There are three Python packages. Each has physics derived from first principles, published models, and tests against published reference values.

| Package | MDA Space relevance | Highlights |
|---|---|---|
| [`orbitkit`](orbitkit) | Satellite operations, constellations (Telesat Lightspeed, AURORA) | SGP4 matching Vallado's verification vectors to 1 mm, TEME→PEF, J2, pass prediction, Walker coverage |
| [`spacearm`](spacearm) | Canadarm2 / Canadarm3, RPO | 7-DOF SSRMS-like DH kinematics, damped least squares IK with redundancy, Clohessy-Wiltshire guidance, free-flyer capture simulation |
| [`linkbudget`](linkbudget) | Ka-band payloads (Sainte-Anne-de-Bellevue) | ITU-R P.618-13 / P.838-3 rain fade, G/T, DVB-S2 ACM, Doppler over a real pass |

```bash
for p in orbitkit spacearm linkbudget; do (cd $p && python -m pytest -q); done
```
