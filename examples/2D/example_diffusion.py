"""
example_2d_diffusion.py
=======================
Simple 2D diffusion example using the HNGD class framework.

Solves hydrogen transport (Fick diffusion only) on a unit square [0,1]^2:
  - Left boundary  (x=0): C_SS = 250 wt.ppm  (hydrogen source)
  - Right boundary (x=1): C_SS = 0   wt.ppm  (hydrogen sink)
  - Top/Bottom           : zero flux (natural BC)

Initial condition: C_SS = 0 everywhere, C_prec = 0 everywhere.

Temperature T = 550 K uniform. At this temperature:
  TSS_D ~ 41.5 wt.ppm, TSS_P ~ 3.4e3 wt.ppm
Since C_SS starts at 0 and rises toward TSS_D from the left BC,
the dissolution source fires briefly near the left wall but the
dominant physics is Fick diffusion from left to right.
C_prec = 0 everywhere so growth never triggers.

This example demonstrates:
  - Using HNGDMaterial, HNGDSource, HNGDProblem on a 2D mesh
  - Setting Dirichlet BCs on a mixed function space subspace
  - Spatial variation of C_SS across the domain
  - Post-processing nodal data from the mixed function space

Run:
    python example_2d_diffusion.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
from dolfinx.fem import locate_dofs_geometrical, dirichletbc
from mpi4py import MPI
from petsc4py import PETSc

import sys
sys.path.append('../../hngd')

from hngd_material import HNGDMaterial
from hngd_source   import HNGDSource
from hngd_problem  import HNGDProblem

# =============================================================================
# Problem parameters
# =============================================================================

T_val   = 550.0    # K  — uniform temperature
C_left  = 1   # wt.ppm — fixed concentration at left wall
C_right = 0.0      # wt.ppm — fixed concentration at right wall
t_final = 1000.0   # s
dt      = 50.0     # s

# =============================================================================
# Mesh — 2D unit square
# =============================================================================

nx, ny = 20, 20
mesh = dmesh.create_unit_square(MPI.COMM_WORLD, nx, ny,
                                 dmesh.CellType.quadrilateral)

# =============================================================================
# Temperature field
# =============================================================================

V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
T = fem.Function(V_scalar)
T.x.array[:] = T_val

# =============================================================================
# Material and source
# =============================================================================

mat = HNGDMaterial(
    D0    = 1e-3,
    E_D   = 0.0,
    KD0   = 0.0,   # no dissolution
    KN0   = 0.0,   # no nucleation
    Kth0  = 0.0,   # no growth (reaction-controlled)
    Kmob0 = 0.0,   # no growth (diffusion-controlled)
)
source = HNGDSource(mat, T, KG_override=0.0)

# =============================================================================
# Initial conditions — zero hydrogen everywhere
# =============================================================================

ic = {'C_SS': 0.0, 'C_prec': 0.0}

# =============================================================================
# Build problem first to get access to V, V0, V0_map for BC construction
# =============================================================================

# Stage 1: function space + ICs — V, V0, V1 available immediately
problem = HNGDProblem(mesh, source, initial_conditions=ic)

# =============================================================================
# Boundary conditions — built using problem.V and problem.V0 from Stage 1
# =============================================================================

def left_wall(x):
    return np.isclose(x[0], 0.0)

def right_wall(x):
    return np.isclose(x[0], 1.0)

c_left_const  = fem.Constant(mesh, PETSc.ScalarType(C_left))
c_right_const = fem.Constant(mesh, PETSc.ScalarType(C_right))

dofs_left  = locate_dofs_geometrical((problem.V.sub(0), problem.V0), left_wall)
dofs_right = locate_dofs_geometrical((problem.V.sub(0), problem.V0), right_wall)

bc_left  = dirichletbc(c_left_const,  dofs_left[0],  problem.V.sub(0))
bc_right = dirichletbc(c_right_const, dofs_right[0], problem.V.sub(0))

# Stage 2: assemble weak form and solver with correct BCs
problem.setup(bcs=[bc_left, bc_right], include_soret=False)

# =============================================================================
# Run
# =============================================================================

print(f"Running 2D diffusion example")
print(f"  Mesh      : {nx}x{ny} quad elements on unit square")
print(f"  T         : {T_val} K (uniform)")
print(f"  BCs       : C_SS={C_left} wt.ppm (left),  C_SS={C_right} wt.ppm (right)")
print(f"  t_final   : {t_final} s,  dt = {dt} s")
print()

times, CSS_t, Cp_t = problem.run(t_final=t_final, dt=dt, verbose=True,
                                  export_bp="example_2d_diffusion.bp")

# =============================================================================
# Post-processing — extract 2D nodal field at final time
# =============================================================================

# Get mesh coordinates and C_SS DOF values
# For CG1 on a 2D mesh, DOF i corresponds to node i
V0_collapsed = problem.V0

# Nodal coordinates from the collapsed subspace
x_coords = V0_collapsed.tabulate_dof_coordinates()   # shape (n_dofs, 3)
x_node   = x_coords[:, 0]   # x-coordinate of each DOF
y_node   = x_coords[:, 1]   # y-coordinate of each DOF

# C_SS values at final time (indexed by V0_map)
CSS_nodal = problem.u.x.array[problem.V0_map]
Cp_nodal  = problem.u.x.array[problem.V1_map]

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# --- Spatial map of C_SS at final time ---
ax = axes[0]
sc = ax.scatter(x_node, y_node, c=CSS_nodal, cmap='viridis',
                s=15, vmin=0, vmax=C_left)
plt.colorbar(sc, ax=ax, label=r'$C_{SS}$ [wt.ppm]')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"$C_{{SS}}$ at t={t_final:.0f}s\n"
             f"T={T_val}K, left={C_left}, right={C_right} wt.ppm")
ax.set_aspect('equal')

# --- Axial profile: C_SS vs x averaged over y ---
# Group by x-coordinate and average C_SS
x_unique = np.unique(np.round(x_node, 8))
CSS_profile = np.array([
    np.mean(CSS_nodal[np.isclose(x_node, xv)]) for xv in x_unique
])

# Steady-state analytical: linear C_SS = C_left * (1 - x)
x_analytical = np.linspace(0, 1, 200)
C_ss_analytical = C_left * (1.0 - x_analytical) + C_right * x_analytical

ax = axes[1]
ax.plot(x_analytical, C_ss_analytical, 'k--', lw=2, label='Steady state (linear)')
ax.plot(x_unique, CSS_profile, 'o-', color='tab:blue', ms=5,
        label=f'DOLFINx t={t_final:.0f}s')
ax.set_xlabel("x")
ax.set_ylabel(r"$\langle C_{SS} \rangle_y$ [wt.ppm]")
ax.set_title("Axial profile (y-averaged)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Domain-averaged C_SS vs time ---
ax = axes[2]
ax.plot(times, CSS_t, 'o-', color='tab:blue', ms=4, label=r'$\langle C_{SS} \rangle$')
ax.plot(times, Cp_t,  's-', color='tab:orange', ms=4, label=r'$\langle C_{prec} \rangle$')
ax.plot(times, CSS_t + Cp_t, 'k:', lw=1.5, label=r'$C_{SS}+C_{prec}$')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Domain-averaged concentration [wt.ppm]")
ax.set_title("Transient domain averages")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("example_2d_diffusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved to example_2d_diffusion.png")

# =============================================================================
# Quick check: profile should approach linear at late time
# =============================================================================

profile_error = np.max(np.abs(CSS_profile - (C_left * (1.0 - x_unique))))
print(f"Max deviation from linear steady state: {profile_error:.4f} wt.ppm")
print(f"(Expected to be small but nonzero at t={t_final}s — "
      f"approaches zero as t -> infinity)")