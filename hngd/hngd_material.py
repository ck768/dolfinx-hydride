"""
hngd_material.py
================
Pure Python dataclass holding all HNGD material parameters.
No DOLFINx, no UFL, no mesh — just physical constants.

All parameters from Table 1, Passelaigue et al.,
J. Nucl. Mater. 544 (2021) 152683.

Usage
-----
    from hngd_material import HNGDMaterial

    # Default Zircaloy-4 cladding parameters (Table 1)
    mat = HNGDMaterial()

    # Override specific parameters for a different material system
    # e.g. ZrH moderator (Kendrick & Forget 2025)
    mat = HNGDMaterial(
        D0  = 1.53e-7,     # m^2/s (= 1.53e-3 cm^2/s, Majer et al.)
        E_D = 14000 / 96485 * 8.617333e-5 * 96485,  # from J/mol to eV
    )
"""

from dataclasses import dataclass, field


@dataclass
class HNGDMaterial:
    """
    HNGD material parameters — pure scalars, no DOLFINx dependency.

    All parameters are keyword arguments with Table 1 defaults.
    Override any parameter at instantiation to adapt to a different
    material system without subclassing.

    Parameters (Table 1, Passelaigue et al. 2021)
    -----------------------------------------------
    R_eV   : Boltzmann constant [eV/K]

    TSS_D0 : TSS_D pre-exponential [wt.ppm]
    Q_D    : TSS_D activation energy [eV/atom]
    TSS_P0 : TSS_P pre-exponential [wt.ppm]
    Q_P    : TSS_P activation energy [eV/atom]

    D0     : Diffusion pre-exponential [m^2/s]
    E_D    : Diffusion activation energy [eV/atom]
    Q_star : Heat of transport for Soret effect [eV/atom]

    KD0    : Dissolution rate pre-exponential [s^-1]
    KN0    : Nucleation rate pre-exponential [s^-1]
    Kth0   : Reaction-controlled growth pre-exponential [s^-1]
    Kmob0  : Diffusion-controlled growth pre-exponential [s^-1]
    EG     : Growth activation energy [eV/atom]

    Eth0..3 : Formation energy polynomial coefficients (Eq. 9)
              E_th(T) = -Eth0 + Eth1*T - Eth2*T^2 + Eth3*T^3  [eV/atom]

    p_jmak : JMAK growth exponent (2.5 for platelets)

    Notes
    -----
    f_alpha and v0 prefactors (Eqs. 7-8) are currently set to 1
    throughout. Full implementation requires atomic fraction
    conversions via M_H and M_Zr — a future extension.

    K_G from Table 1 gives ~1e-12 s^-1 at 650 K due to the large
    EG = 0.9 eV barrier dominating the harmonic mean. This is
    inconsistent with Fig. 3c. The full Lacroix (2019) parameter
    set (ref [4]) is not reproduced in Table 1. When running the
    growth verification case, pass KG_override directly to
    HNGDSource rather than modifying this class.
    """

    # Physical constant
    R_eV   : float = 8.617333e-5   # eV/K

    # Terminal solid solubility (Eq. 5)
    TSS_D0 : float = 1.02e5        # wt.ppm
    Q_D    : float = 0.37          # eV/atom
    TSS_P0 : float = 3.08e4        # wt.ppm
    Q_P    : float = 0.26          # eV/atom

    # Diffusion coefficient (Arrhenius)
    D0     : float = 1.08e-7       # m^2/s  (= 1.08e-2 cm^2/s)
    E_D    : float = 0.46          # eV/atom

    # Soret heat of transport
    Q_star : float = 0.26          # eV/atom

    # Dissolution kinetics (Eq. 6)
    KD0    : float = 1.11e3        # s^-1

    # Nucleation kinetics (Eq. 7, f_alpha = 1)
    KN0    : float = 2.75e-5       # s^-1

    # Growth kinetics (Eq. 8, harmonic mean, f_alpha = v0 = 1)
    Kth0   : float = 5.35e5        # s^-1  reaction-controlled
    Kmob0  : float = 1.6e-5        # s^-1  diffusion-controlled
    EG     : float = 0.9           # eV/atom

    # Formation energy polynomial (Eq. 9)
    # E_th(T) = -Eth0 + Eth1*T - Eth2*T^2 + Eth3*T^3  [eV/atom]
    Eth0   : float =  5.66e-1      # eV/atom
    Eth1   : float =  4.0e-4       # eV/(atom·K)
    Eth2   : float =  2.0e-7       # eV/(atom·K^2)
    Eth3   : float =  3.0e-10      # eV/(atom·K^3)

    # JMAK growth exponent (2.5 for platelets)
    p_jmak : float = 2.5

    def summary(self):
        """Print a formatted summary of all parameters."""
        print("HNGDMaterial parameters")
        print("=" * 45)
        print(f"  R_eV   = {self.R_eV:.6e}  eV/K")
        print(f"  TSS_D0 = {self.TSS_D0:.3e}  wt.ppm,  Q_D  = {self.Q_D:.2f} eV/atom")
        print(f"  TSS_P0 = {self.TSS_P0:.3e}  wt.ppm,  Q_P  = {self.Q_P:.2f} eV/atom")
        print(f"  D0     = {self.D0:.3e}  m^2/s,  E_D  = {self.E_D:.2f} eV/atom")
        print(f"  Q_star = {self.Q_star:.2f}  eV/atom")
        print(f"  KD0    = {self.KD0:.3e}  s^-1")
        print(f"  KN0    = {self.KN0:.3e}  s^-1")
        print(f"  Kth0   = {self.Kth0:.3e}  s^-1")
        print(f"  Kmob0  = {self.Kmob0:.3e}  s^-1,  EG = {self.EG:.2f} eV/atom")
        print(f"  Eth0..3 = {self.Eth0}, {self.Eth1}, {self.Eth2}, {self.Eth3}")
        print(f"  p_jmak = {self.p_jmak}")
        print("=" * 45)