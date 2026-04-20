"""
heat_problem.py
============
HeatProblem class — steady-state heat conduction.

Solves:
    -div(k * grad(T)) = Q(x)

with optional nonlinear radiative BC at fuel pin / heat pipe interfaces:
    -k * grad(T) . n = sigma * eps * (T_source^4 - T^4)

Two-stage construction matching HNGDProblem:
  Stage 1 — __init__: builds scalar CG1 function space, exposes V and T.
  Stage 2 — setup(bcs, ...): assembles weak form and solver.

Usage
-----
    from hngd_heat import HeatProblem
    import dolfinx.fem as fem
    import dolfinx.mesh as dmesh
    from mpi4py import MPI
    from petsc4py import PETSc

    mesh  = dmesh.create_unit_square(MPI.COMM_WORLD, 20, 20)

    # Heating rate from OpenMC (or analytical for testing)
    Q_func = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))
    Q_func.x.array[:] = 1e6   # W/m^3, uniform

    # Stage 1
    heat = HeatProblem(mesh, k=20.0, Q=Q_func)

    # Build BCs using heat.V
    T_wall = fem.Constant(mesh, PETSc.ScalarType(1000.0))
    dofs   = fem.locate_dofs_geometrical(heat.V, lambda x: np.isclose(x[0], 0.0))
    bc     = fem.dirichletbc(T_wall, dofs, heat.V)

    # Stage 2
    heat.setup(bcs=[bc])
    heat.solve()

    # T available as fem.Function for HNGDSource
    source = HNGDSource(mat, heat.T, KG_override=0.0)
"""

import numpy as np
import dolfinx
import dolfinx.fem as fem
import basix.ufl
import ufl
from ufl import dx, ds, grad, dot, inner, TestFunction
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import VTXWriter
from petsc4py import PETSc


# Stefan-Boltzmann constant
SIGMA_SB = 5.670374419e-8   # W/(m^2 K^4)


class HeatProblem:
    """
    Steady-state heat conduction with optional radiative boundary conditions.

    Solves:
        -div(k(T) * grad(T)) = Q(x)

    Radiative BC on marked facets (e.g. fuel pin surfaces):
        -k * grad(T) . n = sigma * epsilon * (T_source^4 - T^4)

    This is nonlinear in T — NonlinearProblem + SNES handles it correctly,
    same as HNGDProblem.

    Parameters (Stage 1 — __init__)
    ---------------------------------
    mesh : dolfinx.mesh.Mesh
        Any geometry, any dimension.

    k : float or fem.Function
        Thermal conductivity [W/(m·K)].
        float   — uniform, converted to fem.Constant internally.
        fem.Function — spatially or temperature-varying k(x) or k(T).

    Q : fem.Function or fem.Constant or float
        Volumetric heating rate [W/m³].
        Typically interpolated from OpenMC heating tallies.
        float/Constant — uniform heating (for testing).

    Parameters (Stage 2 — setup)
    ------------------------------
    bcs : list of fem.dirichletbc
        Dirichlet BCs on heat.V (scalar space, no subspace needed).
        e.g. fixed T at heat pipe walls.

    radiative_bcs : list of dicts or None
        Each dict defines a radiative (Stefan-Boltzmann) BC on a facet set:
            {
              'facet_tags' : MeshTags object,
              'tag'        : int,           facet tag value
              'T_source'   : float,         source temperature [K]
              'emissivity' : float,         surface emissivity [0-1]
            }
        Adds the term: sigma * eps * (T_source^4 - T^4) to the weak form
        on the marked facets. Nonlinear in T — handled by SNES.

    petsc_options : dict or None
        SNES/KSP options. None uses newtonls + LU defaults.

    Attributes available after Stage 1
    ------------------------------------
    V : scalar CG1 function space
    T : fem.Function — temperature field (filled after solve())

    Attributes available after solve()
    ------------------------------------
    T.x.array : nodal temperature values [K]
    """

    def __init__(self, mesh, k, Q):
        self.mesh = mesh

        # Store k — convert float to Constant
        if isinstance(k, (int, float)):
            self.k = fem.Constant(mesh, PETSc.ScalarType(float(k)))
        else:
            self.k = k   # fem.Function or UFL expression

        # Store Q — convert float to Constant
        if isinstance(Q, (int, float)):
            self.Q = fem.Constant(mesh, PETSc.ScalarType(float(Q)))
        else:
            self.Q = Q

        # Stage 1: function space — V and T available immediately
        self._setup_function_space()

    # =========================================================================
    # Stage 1
    # =========================================================================

    def _setup_function_space(self):
        """
        Build scalar CG1 space for temperature.
        Exposes V (function space) and T (solution Function).
        """
        cg1   = basix.ufl.element("Lagrange", self.mesh.basix_cell(), degree=1)
        self.V = fem.functionspace(self.mesh, cg1)
        self.T = fem.Function(self.V, name="Temperature")

        # Named Function for VTX export
        self.T_export = fem.Function(self.V, name="T")

    # =========================================================================
    # Stage 2
    # =========================================================================

    def setup(self, bcs: list = None,
              radiative_bcs: list = None,
              petsc_options: dict = None):
        """
        Assemble weak form and NonlinearProblem.

        Call after building BCs externally using heat.V.

        Parameters
        ----------
        bcs : list of fem.dirichletbc
            Dirichlet BCs on heat.V (no sub() needed — scalar space).
            e.g. fixed T at heat pipe walls:
                T_wall = fem.Constant(mesh, PETSc.ScalarType(1000.0))
                dofs   = fem.locate_dofs_geometrical(heat.V, marker_fn)
                bc     = fem.dirichletbc(T_wall, dofs, heat.V)

        radiative_bcs : list of dicts or None
            Each dict: {'facet_tags': MeshTags, 'tag': int,
                        'T_source': float, 'emissivity': float}

        petsc_options : dict or None
        """
        self.bcs          = bcs if bcs is not None else []
        self.radiative_bcs = radiative_bcs if radiative_bcs is not None else []

        self._setup_weak_form()
        self._setup_solver(petsc_options)

    def _setup_weak_form(self):
        """
        Steady-state heat conduction weak form.

        Strong form:  -div(k * grad(T)) = Q
        Weak form:    integral(k * grad(T) . grad(v)) dx
                    - integral(Q * v) dx
                    + radiative terms on marked facets
                    = 0

        Radiative BC (Stefan-Boltzmann, nonlinear in T):
            -k * grad(T) . n = sigma * eps * (T_source^4 - T^4)
        Weak contribution:
            + integral(sigma * eps * (T^4 - T_source^4) * v) ds(tag)
        """
        T = self.T
        v = TestFunction(self.V)
        k = self.k
        Q = self.Q

        # Diffusion term (integration by parts, natural BC = zero flux)
        F = inner(k * grad(T), grad(v)) * dx

        # Volumetric heat source
        F -= Q * v * dx

        # Radiative BCs — nonlinear Stefan-Boltzmann on marked facets
        for rbc in self.radiative_bcs:
            ft         = rbc['facet_tags']
            tag        = rbc['tag']
            T_src      = fem.Constant(self.mesh, PETSc.ScalarType(rbc['T_source']))
            eps_r      = fem.Constant(self.mesh, PETSc.ScalarType(rbc['emissivity']))
            sigma_c    = fem.Constant(self.mesh, PETSc.ScalarType(SIGMA_SB))

            # Surface measure on tagged facets
            ds_tagged = ufl.Measure("ds", domain=self.mesh,
                                    subdomain_data=ft,
                                    subdomain_id=tag)

            # Contribution: sigma * eps * (T^4 - T_source^4) * v
            # (positive = heat leaving the surface into the domain)
            F += sigma_c * eps_r * (T**4 - T_src**4) * v * ds_tagged

        self.F = F

    def _setup_solver(self, petsc_options: dict = None):
        """Create NonlinearProblem with BCs and configure SNES."""
        if petsc_options is None:
            use_superlu = PETSc.IntType == np.int64
            sys_petsc   = PETSc.Sys()
            if sys_petsc.hasExternalPackage("mumps") and not use_superlu:
                linear_solver = "mumps"
            elif sys_petsc.hasExternalPackage("superlu_dist"):
                linear_solver = "superlu_dist"
            else:
                linear_solver = "petsc"

            petsc_options = {
                "snes_type"                 : "newtonls",
                "snes_linesearch_type"      : "basic",
                "snes_atol"                 : 1e-10,
                "snes_rtol"                 : 1e-10,
                "snes_max_it"               : 50,
                "snes_divergence_tolerance" : "PETSC_UNLIMITED",
                "ksp_type"                  : "preonly",
                "pc_type"                   : "lu",
                "pc_factor_mat_solver_type" : linear_solver,
            }

        self.problem = NonlinearProblem(
            self.F,
            self.T,
            bcs                  = self.bcs,
            petsc_options        = petsc_options,
            petsc_options_prefix = "heat_",
        )

    # =========================================================================
    # Solve
    # =========================================================================

    def solve(self, verbose: bool = True):
        """
        Solve the steady-state heat equation (single Newton solve).

        After calling this, heat.T is the temperature field as a
        fem.Function, ready to pass directly to HNGDSource:
            source = HNGDSource(mat, heat.T, KG_override=0.0)

        Parameters
        ----------
        verbose : bool
            Print convergence information.
        """
        if not hasattr(self, 'problem'):
            raise RuntimeError(
                "Call heat.setup(bcs=[...]) before heat.solve()")

        self.problem.solve()
        converged = self.problem.solver.getConvergedReason()
        n_iter    = self.problem.solver.getIterationNumber()

        if converged <= 0:
            print(f"  WARNING: Heat solver did not converge, reason={converged}")
        elif verbose:
            print(f"  Heat solve: {n_iter} iterations, converged (reason={converged})")
            T_arr = self.T.x.array
            print(f"  T range: [{T_arr.min():.2f}, {T_arr.max():.2f}] K")

        return converged, n_iter

    def export(self, path: str):
        """
        Write temperature field to a VTX (.bp) file.

        Parameters
        ----------
        path : str
            Output path, e.g. 'results/temperature.bp'
        """
        self.T_export.x.array[:] = self.T.x.array
        with VTXWriter(self.mesh.comm, path,
                       [self.T_export], engine="BP5") as writer:
            writer.write(0.0)
        print(f"  Temperature written to {path}")