"""
test_cases.py
=============
Verification tests for the HNGD DOLFINx implementation.

Three cases from Passelaigue et al., J. Nucl. Mater. 544 (2021) 152683,
Section 3.1.1 / Figure 3a, 3b, 3c:

  test_dissolution() : T=550K, all H initially in hydrides
  test_nucleation()  : T=600K, all H initially in solid solution
  test_growth()      : T=650K, all H initially in solid solution + seed

Each test:
  1. Instantiates HNGDMaterial with appropriate parameters
  2. Creates HNGDSource with temperature field
  3. Creates HNGDProblem with mesh, ICs, BCs
  4. Runs the time loop
  5. Compares to analytical solution
  6. Asserts error below tolerance
  7. Plots C_SS, C_prec, and mass conservation check

Run all tests:
    python test_cases.py

Run a single test:
    python test_cases.py dissolution
    python test_cases.py nucleation
    python test_cases.py growth
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc

from hngd_material import HNGDMaterial
from hngd_source   import HNGDSource
from hngd_problem  import HNGDProblem


# =============================================================================
# Analytical solutions (Eqs. 10, 11, 12 — Passelaigue et al.)
# =============================================================================

def analytical_dissolution(t, T, C_tot):
    """Eq. 10: C_SS(t) = TSS_D * (1 - exp(-K_D * t))"""
    R     = 8.617333e-5
    TSS_D = 1.02e5 * np.exp(-0.37 / (R * T))
    K_D   = 1.11e3 * np.exp(-0.46 / (R * T))
    return TSS_D * (1.0 - np.exp(-K_D * t))

def analytical_nucleation(t, T, C_tot):
    """Eq. 11: C_SS(t) = TSS_P + (C_tot - TSS_P) * exp(-K_N * t)"""
    R     = 8.617333e-5
    Eth   = -5.66e-1 + 4e-4*T - 2e-7*T**2 + 3e-10*T**3
    TSS_P = 3.08e4 * np.exp(-0.26 / (R * T))
    K_N   = 2.75e-5 * np.exp(-Eth / (R * T))
    return TSS_P + (C_tot - TSS_P) * np.exp(-K_N * t)

def analytical_growth(t, T, C_tot, KG, TSS_D):
    """Eq. 12: C_SS(t) = TSS_D + (C_tot - TSS_D) * exp(-(K_G*t)^p)"""
    return TSS_D + (C_tot - TSS_D) * np.exp(-(KG * t)**2.5)


# =============================================================================
# Shared plotting helper
# =============================================================================

def plot_results(case_name, times, CSS_t, Cp_t,
                 C_tot, t_final, analytical_fn, analytical_kw,
                 ylim_CSS, ylim_Cp, T_val, dt_val):
    """Three-panel plot: C_SS, C_prec, mass conservation."""

    t_fine = np.linspace(0, t_final, 2000)
    C_ref  = analytical_fn(t_fine, T_val, C_tot, **analytical_kw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # C_SS
    ax = axes[0]
    ax.plot(t_fine, C_ref, 'k-', lw=2, label='Analytical')
    ax.plot(times, CSS_t, 'o', ms=4, color='tab:blue',
            label=f'DOLFINx (dt={dt_val:.0f}s)')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$C_{SS}$ [wt.ppm]")
    ax.set_title(f"{case_name.capitalize()} — Mobile hydrogen\n"
                 f"T={T_val} K,  $C_{{tot}}$={C_tot} wt.ppm")
    ax.set_ylim(ylim_CSS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # C_prec
    ax = axes[1]
    ax.plot(t_fine, C_tot - C_ref, 'k-', lw=2,
            label=r'Analytical ($C_{tot}-C_{SS}$)')
    ax.plot(times, Cp_t, 's', ms=4, color='tab:orange', label='DOLFINx')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$C_{prec}$ [wt.ppm]")
    ax.set_title("Precipitated hydrogen")
    ax.set_ylim(ylim_Cp)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mass conservation
    ax = axes[2]
    ax.plot(times, CSS_t + Cp_t, '-', color='tab:green', lw=1.5,
            label=r'$C_{SS}+C_{prec}$')
    ax.axhline(C_tot, color='k', ls='--', lw=1,
               label=f"$C_{{tot}}$={C_tot}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total hydrogen [wt.ppm]")
    ax.set_title("Mass conservation check\n(should be flat)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    results_path = "../verification/results"
    fname = f"test_{case_name}.png"
    plt.savefig(f"{results_path}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {fname}")


# =============================================================================
# Test cases
# =============================================================================

def test_dissolution(tol=1.0, dt=2.0, verbose=False):
    """
    Dissolution verification — Figure 3a.
    T = 550 K, C_tot = 250 wt.ppm, C_SS(0) = 0 (all H in hydrides).
    Analytical: C_SS(t) = TSS_D * (1 - exp(-K_D * t))
    """

    print("\n" + "="*60)
    print("  TEST: Dissolution  |  T=550 K  |  dt={:.0f}s".format(dt))
    print("="*60)

    T_val   = 550.0
    C_tot   = 250.0
    t_final = 100.0

    # --- Mesh ---
    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 10)

    # --- Temperature field ---
    # fem.Function on collapsed subspace for UFL compatibility (grad(T) usable)
    # Set uniform value — Soret is inactive (include_soret=False)
    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    T = fem.Function(V_scalar)
    T.x.array[:] = T_val

    # --- Material and source ---
    # No overrides: Table 1 parameters are appropriate for dissolution at 550 K
    mat    = HNGDMaterial()
    source = HNGDSource(mat, T)

    # --- Initial conditions ---
    # All hydrogen starts as hydrides: C_SS = 0, C_prec = C_tot
    ic = {'C_SS': 0.0, 'C_prec': C_tot}

    # --- Boundary conditions ---
    # No-flux (natural BC): uniform domain, pure kinetics, no spatial gradient
    bcs = []

    # --- Problem ---
    problem = HNGDProblem(
        mesh               = mesh,
        source             = source,
        initial_conditions = ic,
        bcs                = bcs,
        include_soret      = False,
    )

    # --- Run ---
    times, CSS_t, Cp_t = problem.run(t_final=t_final, dt=dt, verbose=verbose)

    # --- Check ---
    C_ref_final = analytical_dissolution(times[-1], T_val, C_tot)
    error       = abs(CSS_t[-1] - C_ref_final)
    mass_err    = abs(CSS_t[-1] + Cp_t[-1] - C_tot)

    print(f"  Final C_SS      : {CSS_t[-1]:.6f} wt.ppm")
    print(f"  Analytical C_SS : {C_ref_final:.6f} wt.ppm")
    print(f"  Error           : {error:.4e} wt.ppm")
    print(f"  Mass cons. error: {mass_err:.4e} wt.ppm")

    assert error < tol, (
        f"Dissolution error {error:.4e} exceeds tolerance {tol} wt.ppm")
    assert mass_err < 1e-6, (
        f"Mass conservation error {mass_err:.4e} exceeds 1e-6 wt.ppm")

    print("  PASSED")

    plot_results(
        case_name     = 'dissolution',
        times=times, CSS_t=CSS_t, Cp_t=Cp_t,
        C_tot=C_tot, t_final=t_final,
        analytical_fn = analytical_dissolution,
        analytical_kw = {},
        ylim_CSS      = (0, 50),
        ylim_Cp       = (200, 260),
        T_val=T_val, dt_val=dt,
    )

    return times, CSS_t, Cp_t


def test_nucleation(tol=5.0, dt=10.0, verbose=False):
    """
    Nucleation verification — Figure 3b.
    T = 600 K, C_tot = 540 wt.ppm, C_SS(0) = 540 (all H in solution).
    Analytical: C_SS(t) = TSS_P + (C_tot - TSS_P) * exp(-K_N * t)
    """

    print("\n" + "="*60)
    print("  TEST: Nucleation  |  T=600 K  |  dt={:.0f}s".format(dt))
    print("="*60)

    T_val   = 600.0
    C_tot   = 540.0
    t_final = 400.0

    # --- Mesh ---
    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 10)

    # --- Temperature field ---
    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    T = fem.Function(V_scalar)
    T.x.array[:] = T_val

    # --- Material and source ---
    # No overrides: Table 1 parameters appropriate for nucleation at 600 K
    mat    = HNGDMaterial()
    source = HNGDSource(mat, T)

    # --- Initial conditions ---
    # All hydrogen in solid solution: C_SS = C_tot, C_prec = 0
    ic = {'C_SS': C_tot, 'C_prec': 0.0}

    # --- Boundary conditions ---
    bcs = []

    # --- Problem ---
    problem = HNGDProblem(
        mesh               = mesh,
        source             = source,
        initial_conditions = ic,
        bcs                = bcs,
        include_soret      = False,
    )

    # --- Run ---
    times, CSS_t, Cp_t = problem.run(t_final=t_final, dt=dt, verbose=verbose)

    # --- Check ---
    C_ref_final = analytical_nucleation(times[-1], T_val, C_tot)
    error       = abs(CSS_t[-1] - C_ref_final)
    mass_err    = abs(CSS_t[-1] + Cp_t[-1] - C_tot)

    print(f"  Final C_SS      : {CSS_t[-1]:.6f} wt.ppm")
    print(f"  Analytical C_SS : {C_ref_final:.6f} wt.ppm")
    print(f"  Error           : {error:.4e} wt.ppm")
    print(f"  Mass cons. error: {mass_err:.4e} wt.ppm")

    assert error < tol, (
        f"Nucleation error {error:.4e} exceeds tolerance {tol} wt.ppm")
    assert mass_err < 1e-6, (
        f"Mass conservation error {mass_err:.4e} exceeds 1e-6 wt.ppm")

    print("  PASSED")

    plot_results(
        case_name     = 'nucleation',
        times=times, CSS_t=CSS_t, Cp_t=Cp_t,
        C_tot=C_tot, t_final=t_final,
        analytical_fn = analytical_nucleation,
        analytical_kw = {},
        ylim_CSS      = (150, 560),
        ylim_Cp       = (0, 400),
        T_val=T_val, dt_val=dt,
    )

    return times, CSS_t, Cp_t


def test_growth(tol=5.0, dt=50.0, verbose=False):
    """
    Growth verification — Figure 3c.
    T = 650 K, C_tot = 288 wt.ppm, C_SS(0) = 288 (all H in solution).
    C_prec seeded at 1e-6 wt.ppm to trigger JMAK (Section 3.1.1).

    KG_override and TSS_D_override are passed here — not in the material
    class — because they are calibration values specific to this test case
    to compensate for the incomplete Table 1 parameters (Lacroix 2019 ref [4]).
    TSS_D = 144 wt.ppm is stated explicitly in the paper for this case.
    K_G = 1.71e-4 s^-1 is back-calculated from Fig. 3c.

    Analytical: C_SS(t) = TSS_D + (C_tot - TSS_D) * exp(-(K_G*t)^2.5)
    """

    print("\n" + "="*60)
    print("  TEST: Growth  |  T=650 K  |  dt={:.0f}s".format(dt))
    print("="*60)

    T_val   = 650.0
    C_tot   = 288.0
    t_final = 8000.0

    # Growth-specific calibration values
    KG_val   = 1.71e-4   # s^-1  — calibrated to Fig. 3c
    TSS_D_val = 144.0    # wt.ppm — paper states explicitly for 650 K

    # --- Mesh ---
    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 10)

    # --- Temperature field ---
    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    T = fem.Function(V_scalar)
    T.x.array[:] = T_val

    # --- Material ---
    # Default Table 1 parameters — K_G and TSS_D are overridden in HNGDSource
    mat = HNGDMaterial()

    # --- Source ---
    # KG_override and TSS_D_override live here, not in HNGDMaterial,
    # because they are case-specific calibration values, not material properties
    source = HNGDSource(
        material       = mat,
        T              = T,
        KG_override    = KG_val,
        TSS_D_override = TSS_D_val,
    )

    # --- Initial conditions ---
    # All H in solid solution, tiny C_prec seed to trigger JMAK growth
    ic = {'C_SS': C_tot, 'C_prec': 1e-6}

    # --- Boundary conditions ---
    bcs = []

    # --- Problem ---
    problem = HNGDProblem(
        mesh               = mesh,
        source             = source,
        initial_conditions = ic,
        bcs                = bcs,
        include_soret      = False,
    )

    # --- Run ---
    times, CSS_t, Cp_t = problem.run(t_final=t_final, dt=dt, verbose=verbose)

    # --- Check ---
    C_ref_final = analytical_growth(times[-1], T_val, C_tot,
                                    KG=KG_val, TSS_D=TSS_D_val)
    seed = 1e-6
    error    = abs(CSS_t[-1] - C_ref_final)
    mass_err = abs(CSS_t[-1] + Cp_t[-1] - C_tot - seed)

    print(f"  Final C_SS      : {CSS_t[-1]:.6f} wt.ppm")
    print(f"  Analytical C_SS : {C_ref_final:.6f} wt.ppm")
    print(f"  Error           : {error:.4e} wt.ppm")
    print(f"  Mass cons. error: {mass_err:.4e} wt.ppm")

    assert error < tol, (
        f"Growth error {error:.4e} exceeds tolerance {tol} wt.ppm")
    assert mass_err < 1e-6, (
        f"Mass conservation error {mass_err:.4e} exceeds 1e-6 wt.ppm")

    print("  PASSED")

    plot_results(
        case_name     = 'growth',
        times=times, CSS_t=CSS_t, Cp_t=Cp_t,
        C_tot=C_tot, t_final=t_final,
        analytical_fn = analytical_growth,
        analytical_kw = {'KG': KG_val, 'TSS_D': TSS_D_val},
        ylim_CSS      = (130, 300),
        ylim_Cp       = (0, 160),
        T_val=T_val, dt_val=dt,
    )

    return times, CSS_t, Cp_t


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # Select which tests to run from command line, or run all
    available = {
        'dissolution': test_dissolution,
        'nucleation' : test_nucleation,
        'growth'     : test_growth,
    }

    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        for name in requested:
            if name not in available:
                print(f"Unknown test: '{name}'. "
                      f"Choose from: {list(available.keys())}")
                sys.exit(1)
        tests_to_run = {k: available[k] for k in requested}
    else:
        tests_to_run = available

    passed = []
    failed = []

    for name, fn in tests_to_run.items():
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    print("\n" + "="*60)
    print(f"  Results: {len(passed)} passed, {len(failed)} failed")
    if passed:
        print(f"  Passed : {passed}")
    if failed:
        print(f"  Failed : {failed}")
        sys.exit(1)