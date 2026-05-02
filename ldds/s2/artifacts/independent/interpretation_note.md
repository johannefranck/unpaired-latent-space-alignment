# Interpretation note: S2 LDD/coupling control

On S2, the intrinsic curvature is constant. Hence pointwise LDD variation is not expected to reflect point-dependent curvature variation. In this control experiment, the LDD signal mainly reflects sampling density variation.

Mean values across distance types and repetitions:

## Uniform
- LDD variance: 1.435584e-05
- Coupling response E_R: 4.984196e-07
- Coupling peak ratio: 1.058225

## One-mode vMF
- LDD variance: 4.228528e-03
- LDD effective rank: 1.208302
- LDD spectral entropy: 0.189104
- Coupling response E_R: 3.158830e-03
- Coupling peak ratio: 10.354271

## Mixture-vMF
- LDD variance: 3.956424e-04
- LDD effective rank: 2.038352
- LDD spectral entropy: 0.709883
- Coupling response E_R: 6.098768e-04
- Coupling peak ratio: 4.454122

## Suggested write-up

On S2, the LDD signatures form a density-dominated control case. Since the sphere has constant curvature, pointwise LDD variation primarily reflects variation in the sampling density rather than variation in curvature. The uniform distribution gives near-zero LDD variance and near-uniform couplings up to finite-sample noise. The one-mode vMF produces the strongest LDD variance and the strongest coupling response, because its density has a large smooth gradient. The mixture-vMF can exhibit higher LDD effective rank and spectral entropy, indicating richer LDD variation, but this does not necessarily translate into sharper or more non-uniform couplings. Thus, in this S2 control, coupling response is driven more by LDD amplitude than by LDD spectral complexity.
