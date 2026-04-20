"""
hngd_problem.py
===============
HNGDProblem class — owns all DOLFINx infrastructure.

Two-stage construction pattern:
  Stage 1 — __init__: builds function space and applies ICs.
             V, V0, V1, V0_map, V1_map, u, u_n are available immediately.
             Use these to build Dirichlet BCs externally (Option A).

  Stage 2 — setup(bcs, petsc_options, include_soret):
             Assembles weak form and creates NonlinearProblem with correct BCs.
             Must be called before run() or step().

Usage
-----
    from hngd_material import HNGDMaterial
    from hngd_source   import HNGDSource
    from hngd_problem  import HNGDProblem
    import dolfinx.fem as fem
    import dolfinx.mesh as dmesh
    from mpi4py import MPI
    from petsc4py import PETSc

    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 20, 20)

    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    T = fem.Function(V_scalar)
    T.x.array[:] = 550.0

    mat    = HNGDMaterial()
    source = HNGDSource(mat, T)

    # Stage 1: function space + ICs — V, V0, V1 available now
    problem = HNGDProblem(mesh, source, initial_conditions={'C_SS': 0.0, 'C_prec': 0.0})

    # Build BCs using exposed problem.V, problem.V0
    c_left = fem.Constant(mesh, PETSc.ScalarType(250.0))
    dofs   = fem.locate_dofs_geometrical((problem.V.sub(0), problem.V0), left_wall)
    bc     = fem.dirichletbc(c_left, dofs[0], problem.V.sub(0))

    # Stage 2: assemble weak form and solver with correct BCs
    problem.setup(bcs=[bc])

    # Run — optional VTX export
    problem.run(t_final=100.0, dt=2.0, export_bp='results/output.bp')
"""

import numpy as np
import dolfinx
import dolfinx.fem as fem
import basix.ufl
import ufl
from ufl import dx, grad, dot, split, TestFunctions
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

from hngd_source import HNGDSource


class HNGDProblem:
    """
    Two-field HNGD problem: C_SS (mobile) + C_prec (precipitate).

    Solves:
        dC_SS/dt   - div(D*grad(C_SS) + Soret) = S_SS
        dC_prec/dt                              = -S_SS

    Two-stage construction — separates function space setup (Stage 1)
    from weak form assembly (Stage 2) so BCs can be built externally
    using the exposed function space attributes before finalisation.

    Parameters (Stage 1 — __init__)
    ---------------------------------
    mesh : dolfinx.mesh.Mesh
        Any geometry, any dimension.
    source : HNGDSource
        Pre-built source. Its T field must live on this mesh.
    initial_conditions : dict
        {'C_SS': float or callable, 'C_prec': float or callable}
        float   — uniform value across all DOFs
        callable — passed to fem.Function.interpolate for spatial variation

    Parameters (Stage 2 — setup)
    ------------------------------
    bcs : list of fem.dirichletbc
        Pre-built BCs on V.sub(0) for C_SS and/or V.sub(1) for C_prec.
        Build after Stage 1 using problem.V, problem.V0, problem.V1:
            dofs = fem.locate_dofs_geometrical((problem.V.sub(0), problem.V0), marker)
            bc   = fem.dirichletbc(value, dofs[0], problem.V.sub(0))
        Pass [] for no-flux (natural BC) — verification cases.
    petsc_options : dict or None
        SNES/KSP options. None uses FESTIM-template defaults (newtonls + LU).
    include_soret : bool
        Add Soret term. Requires source.T to be a fem.Function — grad() of
        a fem.Constant is undefined in UFL and raises an error.

    Attributes available after Stage 1
    ------------------------------------
    V, V0, V1          : mixed and collapsed function spaces
    V0_map, V1_map     : DOF index maps for C_SS and C_prec
    u, u_n             : current and previous solution Functions
    CSS_func, Cp_func  : named collapsed Functions used for VTX export

    Attributes available after run()
    ---------------------------------
    times, CSS_t, Cp_t : time history of domain-averaged values
    """

    def __init__(self, mesh, source: HNGDSource, initial_conditions: dict):

        self.mesh   = mesh
        self.source = source

        self.times = []
        self.CSS_t = []
        self.Cp_t  = []

        # Stage 1 — immediately available for BC construction
        self._setup_function_space()
        self._apply_initial_conditions(initial_conditions)

    # =========================================================================
    # Stage 1
    # =========================================================================

    def _setup_function_space(self):
        """
        Build mixed CG1 space [C_SS, C_prec].
        Exposes V, V0, V1, V0_map, V1_map, u, u_n, CSS_func, Cp_func.
        """
        cg1        = basix.ufl.element("Lagrange", self.mesh.basix_cell(), degree=1)
        mixed_elem = basix.ufl.mixed_element([cg1, cg1])
        self.V     = fem.functionspace(self.mesh, mixed_elem)

        self.u   = fem.Function(self.V)
        self.u_n = fem.Function(self.V)

        # UFL split — used in weak form
        self.C_SS,   self.C_prec   = split(self.u)
        self.C_SS_n, self.C_prec_n = split(self.u_n)
        self.v_CSS, self.v_Cprec   = TestFunctions(self.V)

        # Collapsed subspaces — use for BC construction and DOF access
        self.V0, self.V0_map = self.V.sub(0).collapse()
        self.V1, self.V1_map = self.V.sub(1).collapse()

        # Named collapsed Functions for VTX export
        # These are synced from u.x.array before each write
        self.CSS_func = fem.Function(self.V0, name="C_SS")
        self.Cp_func  = fem.Function(self.V1, name="C_prec")

    def _apply_initial_conditions(self, ic: dict):
        """
        Set ICs on both u and u_n.

        ic values:
          float    — uniform scalar broadcast to all DOFs
          callable — passed to fem.Function.interpolate for spatial ICs
                     e.g. lambda x: 250.0 * np.ones(x.shape[1])
        """
        for key, subspace, idx_map in [
            ('C_SS',   self.V0, self.V0_map),
            ('C_prec', self.V1, self.V1_map),
        ]:
            val = ic.get(key, 0.0)
            if callable(val):
                f_tmp = fem.Function(subspace)
                f_tmp.interpolate(val)
                self.u.x.array[idx_map]   = f_tmp.x.array
                self.u_n.x.array[idx_map] = f_tmp.x.array
            else:
                self.u.x.array[idx_map]   = float(val)
                self.u_n.x.array[idx_map] = float(val)

        self.u.x.scatter_forward()
        self.u_n.x.scatter_forward()

    # =========================================================================
    # Stage 2 — call after building BCs externally
    # =========================================================================

    def setup(self, bcs: list = None,
              petsc_options: dict = None,
              include_soret: bool = False,
              steady_state: bool = False):
        """
        Assemble weak form and NonlinearProblem with the provided BCs.

        This is Stage 2 — must be called before run() or step().
        BCs are incorporated correctly into the residual from the start.

        Parameters
        ----------
        bcs : list of fem.dirichletbc or []
        petsc_options : dict or None
        include_soret : bool
        """
        self.bcs           = bcs if bcs is not None else []
        self.include_soret = include_soret
        self.steady_state  = steady_state

        self._setup_weak_form()
        self._setup_solver(petsc_options)

    def _setup_weak_form(self):
        """
        Monolithic weak form F for both fields.

        Mobile (C_SS):
            (C_SS - C_SS_n)/dt * v_CSS * dx
          + D * grad(C_SS) . grad(v_CSS) * dx
          [+ Soret if include_soret]
          - S_SS * v_CSS * dx

        Precipitate (C_prec):
            (C_prec - C_prec_n)/dt * v_Cprec * dx
          + S_SS * v_Cprec * dx

        S_SS enters with opposite signs — mass conservation by construction.
        """
        C_SS     = self.C_SS
        C_prec   = self.C_prec
        C_SS_n   = self.C_SS_n
        C_prec_n = self.C_prec_n
        v_CSS    = self.v_CSS
        v_Cprec  = self.v_Cprec

        # dt as a Constant — update via self.dt.value in the time loop
        self.dt = fem.Constant(self.mesh, PETSc.ScalarType(1.0))

        S_SS = self.source.compute(C_SS, C_prec_n)
        D    = self.source.D
        T    = self.source.T

        # Transient or steady-state
        if self.steady_state:
            # Drop time derivatives — solve for equilibrium H distribution
            # Driven purely by diffusion + Soret + HNGD source at steady state
            F  = fem.Constant(self.mesh, PETSc.ScalarType(0.0)) * v_CSS   * dx
            F += fem.Constant(self.mesh, PETSc.ScalarType(0.0)) * v_Cprec * dx
        else:
            # Backward Euler transient
            F  = (C_SS   - C_SS_n)   / self.dt * v_CSS   * dx
            F += (C_prec - C_prec_n) / self.dt * v_Cprec * dx

        # Fick diffusion on C_SS
        F += D * dot(grad(C_SS), grad(v_CSS)) * dx

        # Soret — only valid when T is a fem.Function (has spatial grad)
        if self.include_soret:
            m = self.source.material
            F += (D * (m.Q_star / (m.R_eV * T**2))
                  * C_SS * dot(grad(T), grad(v_CSS)) * dx)

        # HNGD source — equal and opposite
        F -= S_SS * v_CSS   * dx    # dC_SS/dt   = +S_SS
        F += S_SS * v_Cprec * dx    # dC_prec/dt = -S_SS

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
                "snes_linesearch_type"      : "none",
                "snes_atol"                 : 1e-10,
                "snes_rtol"                 : 1e-10,
                "snes_max_it"               : 100,
                "snes_divergence_tolerance" : "PETSC_UNLIMITED",
                "ksp_type"                  : "preonly",
                "pc_type"                   : "lu",
                "pc_factor_mat_solver_type" : linear_solver,
            }

        self.problem = NonlinearProblem(
            self.F,
            self.u,
            bcs                  = self.bcs,
            petsc_options        = petsc_options,
            petsc_options_prefix = "hngd_",
        )

    # =========================================================================
    # Time stepping
    # =========================================================================

    def step(self, verbose: bool = True):
        """
        Advance one timestep: solve, clip, advance time level.

        Returns (converged_reason, n_iterations).
        Can be called manually for custom time loops with adaptive dt,
        transient BCs, or coupling to external solvers.
        """
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason()
        n_iter    = self.problem.solver.getIterationNumber()

        if converged <= 0:
            print(f"  WARNING: SNES did not converge, reason={converged}")

        # Physical bounds — clip independently (small numerical overshoot near zero)
        self.u.x.array[self.V0_map] = np.clip(
            self.u.x.array[self.V0_map], 0.0, np.inf)
        self.u.x.array[self.V1_map] = np.clip(
            self.u.x.array[self.V1_map], 0.0, np.inf)

        # Advance time level
        self.u_n.x.array[:] = self.u.x.array[:]

        return converged, n_iter

    def _sync_export_functions(self):
        """
        Copy nodal values from u into the named collapsed Functions.
        Called before each VTX write so ParaView sees current field values.
        """
        self.CSS_func.x.array[:] = self.u.x.array[self.V0_map]
        self.Cp_func.x.array[:]  = self.u.x.array[self.V1_map]

    def run(self, t_final: float, dt: float,
            verbose: bool = True,
            export_bp: str = None,
            callback=None):
        """
        Run the time loop from t=0 to t=t_final.

        Parameters
        ----------
        t_final : float
            Simulation end time [s].

        dt : float
            Timestep [s]. Update mid-run via problem.dt.value = new_dt.

        verbose : bool
            Print convergence and field averages each step.

        export_bp : str or None
            Path for VTX output directory, e.g. 'results/output.bp'.
            Writes C_SS and C_prec as named fields at t=0 and every step.
            Follows the same VTXWriter pattern as the ALE example.
            Set None to skip file output.

        callback : callable or None
            Called after each step: callback(problem, t).
            Use for adaptive dt, transient BCs, or updating T:
                def cb(prob, t):
                    prob.source.T.x.array[:] = new_temperature(t)
                    prob.dt.value = new_dt(t)

        Returns
        -------
        times  : np.ndarray  [s]
        CSS_t  : np.ndarray  [wt.ppm]   domain-averaged C_SS
        Cp_t   : np.ndarray  [wt.ppm]   domain-averaged C_prec
        """
        if not hasattr(self, 'problem'):
            raise RuntimeError(
                "Call problem.setup(bcs=[...]) before problem.run()")

        self.dt.value = dt
        self.times = [0.0]
        self.CSS_t = [float(np.mean(self.u.x.array[self.V0_map]))]
        self.Cp_t  = [float(np.mean(self.u.x.array[self.V1_map]))]

        # --- VTX writer ---
        # Opens once, writes at t=0 and after every step, closes at end.
        # CSS_func and Cp_func are named collapsed Functions — ParaView
        # will display them as "C_SS" and "C_prec" in the field selector.
        writer = None
        if export_bp is not None:
            writer = VTXWriter(
                self.mesh.comm,
                export_bp,
                [self.CSS_func, self.Cp_func],
                engine="BP5",
            )
            self._sync_export_functions()
            writer.write(0.0)

        if self.steady_state:
            # Single solve — no time loop
            converged, n_iter = self.step(verbose=verbose)
            t = t_final
            self.times.append(t)
            self.CSS_t.append(float(np.mean(self.u.x.array[self.V0_map])))
            self.Cp_t.append(float(np.mean(self.u.x.array[self.V1_map])))
            if verbose:
                print(f"Steady state | iters={n_iter} | "
                      f"C_SS={self.CSS_t[-1]:.4f}  C_prec={self.Cp_t[-1]:.4f}")
            if writer is not None:
                self._sync_export_functions()
                writer.write(t)
            if callback is not None:
                callback(self, t)
        else:
            t = 0.0
            while t < t_final - 1e-10:
                t += self.dt.value

                converged, n_iter = self.step(verbose=verbose)

                self.times.append(t)
                self.CSS_t.append(float(np.mean(self.u.x.array[self.V0_map])))
                self.Cp_t.append(float(np.mean(self.u.x.array[self.V1_map])))

                if verbose:
                    print(
                        f"t={t:10.2f}s | iters={n_iter} | "
                        f"C_SS={self.CSS_t[-1]:.4f}  "
                        f"C_prec={self.Cp_t[-1]:.4f}  "
                        f"C_tot={self.CSS_t[-1]+self.Cp_t[-1]:.4f}"
                    )

                if writer is not None:
                    self._sync_export_functions()
                    writer.write(t)

                if callback is not None:
                    callback(self, t)

        if writer is not None:
            writer.close()

        self.times = np.array(self.times)
        self.CSS_t = np.array(self.CSS_t)
        self.Cp_t  = np.array(self.Cp_t)

        return self.times, self.CSS_t, self.Cp_t