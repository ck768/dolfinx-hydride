"""
hngd_source.py
==============
HNGDSource class — pure UFL algebra, no mesh or solver infrastructure.

Builds Arrhenius UFL expressions from an HNGDMaterial and a temperature
field once at construction. The compute() method returns S_SS, the net
source on mobile hydrogen, as a UFL expression that drops directly into
any DOLFINx residual.

Usage
-----
    from hngd_material import HNGDMaterial
    from hngd_source   import HNGDSource

    mat    = HNGDMaterial()
    source = HNGDSource(mat, T)          # T: fem.Constant or fem.Function

    S_SS = source.compute(C_SS, C_prec)  # pure UFL expression

    # In weak form:
    F -= S_SS * v_CSS   * dx
    F += S_SS * v_Cprec * dx
"""

import ufl
from ufl import ln
import dolfinx.fem as fem
from petsc4py import PETSc

from hngd_material import HNGDMaterial


class HNGDSource:
    """
    HNGD source term builder — pure UFL, no DOLFINx infrastructure.

    Constructs all Arrhenius expressions from an HNGDMaterial and a
    temperature field at instantiation. These expressions are built once
    and reused every time compute() is called, avoiding redundant symbolic
    construction inside the time loop.

    Parameters
    ----------
    material : HNGDMaterial
        Physical parameters (Table 1, Passelaigue et al. 2021).

    T : fem.Constant or fem.Function
        Temperature field.
        - fem.Constant : uniform T (verification cases). All Arrhenius
          expressions evaluate to scalar UFL constants.
        - fem.Function : spatially varying T (FESTIM-coupled problems).
          All expressions become spatially varying UFL fields.
          Soret term in the weak form (grad(T)) is only meaningful in
          this case.

    KG_override : float or None
        Override for K_G [s^-1]. Pass a calibrated value when Table 1
        parameters are known to be incomplete (e.g. growth verification
        case where Table 1 gives K_G ~ 1e-12 s^-1 at 650 K, inconsistent
        with Fig. 3c). Set to None to use Table 1 Kmob/Kth harmonic mean.

    TSS_D_override : float or None
        Override for TSS_D [wt.ppm]. Paper Section 3.1.1 explicitly states
        TSS_D = 144 wt.ppm at 650 K; Arrhenius gives 137.98 wt.ppm.
        Set to None to use Arrhenius expression.

    Attributes (UFL expressions, built at __init__)
    ------------------------------------------------
    TSS_D, TSS_P : terminal solid solubilities [wt.ppm]
    K_D, K_N, K_G : kinetic rate constants [s^-1]
    D             : diffusion coefficient [m^2/s]
    Eth           : formation energy [eV/atom]

    Methods
    -------
    compute(C_SS, C_prec) -> UFL expression
        Returns S_SS, the net source on mobile hydrogen [wt.ppm/s].
        S_SS > 0 : dissolution (source on C_SS, sink on C_prec)
        S_SS < 0 : nucleation or growth (sink on C_SS, source on C_prec)
    """

    # Numerical guard constants
    _EPS_PREC  = 1e-6    # C_prec seed: prevents JMAK locking to zero at C_prec=0
    _EPS_UPPER = 1e-30   # lower bound on (1-x) to keep ln(1-x) finite

    def __init__(self, material: HNGDMaterial, T,
                 KG_override=None, TSS_D_override=None):

        self.material       = material
        self.T              = T
        self.KG_override    = KG_override
        self.TSS_D_override = TSS_D_override

        self._build_ufl_expressions()

    def _mesh(self):
        """Extract mesh from T whether Constant or Function."""
        if hasattr(self.T, 'function_space'):
            return self.T.function_space.mesh
        return self.T.ufl_domain()

    def _build_ufl_expressions(self):
        """
        Build all Arrhenius UFL expressions from material parameters and T.
        Called once at construction. If T is a fem.Function, all expressions
        are spatially varying fields. If T is a fem.Constant, they are
        constant UFL scalars.
        """
        m = self.material
        T = self.T

        # Formation energy polynomial E_th(T)  (Eq. 9)
        self.Eth = (-m.Eth0
                    + m.Eth1 * T
                    - m.Eth2 * T**2
                    + m.Eth3 * T**3)

        # Terminal solid solubilities  (Eq. 5)
        if self.TSS_D_override is not None:
            self.TSS_D = fem.Constant(
                self._mesh(), PETSc.ScalarType(self.TSS_D_override))
        else:
            self.TSS_D = m.TSS_D0 * ufl.exp(-m.Q_D / (m.R_eV * T))

        self.TSS_P = m.TSS_P0 * ufl.exp(-m.Q_P / (m.R_eV * T))

        # Diffusion coefficient D(T)  (Arrhenius)
        self.D = m.D0 * ufl.exp(-m.E_D / (m.R_eV * T))

        # Dissolution rate K_D  (Eq. 6)
        self.K_D = m.KD0 * ufl.exp(-m.E_D / (m.R_eV * T))

        # Nucleation rate K_N  (Eq. 7, f_alpha = 1)
        self.K_N = m.KN0 * ufl.exp(-self.Eth / (m.R_eV * T))

        # Growth rate K_G  (Eq. 8, harmonic mean, f_alpha = v0 = 1)
        if self.KG_override is not None:
            self.K_G = fem.Constant(
                self._mesh(), PETSc.ScalarType(self.KG_override))
        else:
            K_mob    = m.Kmob0 * ufl.exp(-m.EG       / (m.R_eV * T))
            K_th     = m.Kth0  * ufl.exp(-self.Eth   / (m.R_eV * T))
            self.K_G = 1.0 / (1.0/K_mob + 1.0/K_th)

    def compute(self, C_SS, C_prec):
        """
        Return S_SS — the net HNGD source on mobile hydrogen.

        Parameters
        ----------
        C_SS   : UFL expression
            Mobile hydrogen [wt.ppm], current timestep trial function.
            Dissolution and nucleation are implicit in C_SS.

        C_prec : UFL expression
            Precipitated hydrogen [wt.ppm], previous timestep.
            Used explicitly in the JMAK growth term to avoid the
            negative-Jacobian instability that arises when growth is
            treated implicitly in C_SS alone.

        Returns
        -------
        S_SS : UFL expression  [wt.ppm/s]
            Net source on mobile hydrogen. Add to weak form as:
                F -= S_SS * v_CSS   * dx
                F += S_SS * v_Cprec * dx
            Mass conservation is enforced by construction: the same
            S_SS expression with opposite sign enters both equations.

        Physics
        -------
        Three mechanisms evaluated simultaneously at every quadrature point.
        ufl.conditional activates each term pointwise — different nodes can
        be dissolving, nucleating, and growing in the same timestep.

        Dissolution  (C_SS < TSS_D):
            S_D = -K_D * (C_SS - TSS_D) > 0
            Hydrides dissolve → H enters solid solution.

        Nucleation   (C_SS > TSS_P):
            S_N = -K_N * (C_SS - TSS_P) < 0
            Oversaturated solution → new hydrides nucleate.

        Growth       (C_SS > TSS_D and C_prec > 0, JMAK Eq. 3):
            S_G = -K_G*(C_tot-TSS_D)*p*(1-x)*(-ln(1-x))^(1-1/p) < 0
            Existing hydrides grow by accretion.
            Active in both C_SS > TSS_P (simultaneous with nucleation)
            and TSS_D < C_SS < TSS_P (hysteresis region, HNGD innovation).
        """

        TSS_D = self.TSS_D
        TSS_P = self.TSS_P
        K_D   = self.K_D
        K_N   = self.K_N
        K_G   = self.K_G
        p     = self.material.p_jmak

        eps_p = self._EPS_PREC
        eps_u = self._EPS_UPPER

        C_tot = C_SS + C_prec

        # --- Dissolution ---
        # Active when C_SS < TSS_D: hydrides release H into solid solution
        S_D = ufl.conditional(
            ufl.lt(C_SS, TSS_D),
            -K_D * (C_SS - TSS_D),
            ufl.zero()
        )

        # --- Nucleation ---
        # Active when C_SS > TSS_P: oversaturated solution precipitates
        S_N = ufl.conditional(
            ufl.gt(C_SS, TSS_P),
            -K_N * (C_SS - TSS_P),
            ufl.zero()
        )

        # --- Growth (JMAK, Eq. 3) ---
        # Active when C_SS > TSS_D and existing hydrides present (C_prec > 0)
        # x = C_prec / (C_tot - TSS_D): precipitation advancement [0, 1]
        # eps_p added to prevent x=0 locking the JMAK rate to zero
        # eps_u guards ln(1-x) from diverging at full precipitation
        denom       = C_tot - TSS_D
        x           = (C_prec + eps_p) / denom
        one_minus_x = ufl.max_value(eps_u, 1.0 - x)
        jmak        = p * one_minus_x * (-ln(one_minus_x))**(1.0 - 1.0/p)

        S_G = ufl.conditional(
            ufl.And(
                ufl.gt(C_SS,   TSS_D),
                ufl.gt(C_prec, ufl.zero())
            ),
            -K_G * denom * jmak,
            ufl.zero()
        )

        return S_D + S_N + S_G