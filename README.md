# dolfinx-hydride
Modeling hydrogen transport with hydride formation in Dolfinx


HNGD Model — DOLFINx Setup + Verification Cases
=================================================
Structured to mirror the FESTIM transient template exactly.

Two reusable components:
  HNGDMaterial  — holds all parameters, builds UFL Arrhenius expressions
  HNGD_source   — pure UFL, returns S_SS for the weak form

Verification cases from Passelaigue et al., J. Nucl. Mater. 544 (2021) 152683
Section 3.1.1 / Figure 3a, 3b, 3c:
  'dissolution' : T=550K, C_tot=250, C_SS(0)=0    (all H in hydrides)
  'nucleation'  : T=600K, C_tot=540, C_SS(0)=540  (all H in solution)
  'growth'      : T=650K, C_tot=288, C_SS(0)=288  (all H in solution, seed C_prec)

Select which case to run by setting CASE below.
