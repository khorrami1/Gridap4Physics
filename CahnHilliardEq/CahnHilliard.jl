################################################################################
# Cahn–Hilliard Simulation Using Gridap
#
# This code sets up and solves a transient phase-field model based on the 
# Cahn–Hilliard equation. The simulation uses a finite element method on a 
# two-dimensional unit square domain. The model is discretized in time using 
# the Theta method (with θ = 0.5, corresponding to Crank–Nicolson) combined 
# with Newton’s method for the nonlinear solver. The results (phase field "c" 
# and chemical potential "μ") are written to VTK files for visualization.
################################################################################

using Gridap          # Finite element library for Julia.
using Random          # For reproducibility by setting a random seed.

#------------------------------------------------------------------------------
# Set a fixed random seed for reproducibility of initial condition randomness.
Random.seed!(135791113)

#------------------------------------------------------------------------------
# Define the computational mesh.
# The domain is a unit square [0,1] x [0,1] and it is partitioned into 96×96 cells.
domain = (0.0, 1.0, 0.0, 1.0)     # Domain limits in x and y directions.
partition = (96, 96)              # Number of cells in x and y directions.
model = CartesianDiscreteModel(domain, partition)

#------------------------------------------------------------------------------
# Finite Element Spaces Setup
#
# We define two sets of finite element spaces:
#   - One for the phase field variable "c"
#   - One for the chemical potential "μ"
#
# 'order' specifies the polynomial order (here first order, linear elements).
order = 1

# Define reference finite element (FE) for the phase field.
reffe_c = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe_c, conformity=:H1)   # H1-conforming space for "c".
U = TransientTrialFESpace(V)                  # Transient trial space for time evolution.

# Define reference finite element (FE) for the chemical potential.
reffe_μ = ReferenceFE(lagrangian, Float64, order)
Q = FESpace(model, reffe_μ, conformity=:H1)   # H1-conforming space for "μ".
P = TrialFESpace(Q)                           # Corresponding trial space.

# Create product spaces for the multi-field system (c and μ) in both transient and stationary cases.
X = TransientMultiFieldFESpace([U, P])        # Multi-field transient space.
Y = MultiFieldFESpace([V, Q])                 # Multi-field space for test functions.

#------------------------------------------------------------------------------
# Quadrature and Mesh Measures
#
# The variable `degree` sets the quadrature degree for numerical integration.
degree = 2
Ω = Triangulation(model)                      # Triangulation of the domain.
dΩ = Measure(Ω, degree)                       # Measure used in integrals.

#------------------------------------------------------------------------------
# Model Parameters
#
# M : Mobility coefficient.
# λ : Parameter related to the interfacial energy (gradient term).
M = 1.0
λ = 1.0e-2

#------------------------------------------------------------------------------
# Define the Free Energy Function and its Derivative
#
# The double-well potential f(c) is used and its derivative is needed for the 
# chemical potential.
f(c) = 100*c^2*(1-c)^2                       # Free energy density.
dfdc = (c) -> 100*(4*c*c*c - 6*c*c + 2*c)       # Derivative of f(c) with respect to c.

#------------------------------------------------------------------------------
# Residuals for the Cahn–Hilliard Equations
#
# The weak form of the Cahn–Hilliard equation involves several terms:
#
# 1. Mass term (time derivative term).
# 2. Diffusive term based on the chemical potential gradient.
# 3. Terms capturing the chemical potential's relation to the free energy 
#    derivative and the gradient energy.
#

# Mass residual (for the time derivative of c).
mass(t, (c, μ), (ct, μt), (dc, dμ)) = ∫(ct ⋅ dc) * dΩ
# A variation of the mass residual used for semilinear formulations.
mass_sl(t, (ct, μt), (dc, dμ)) = ∫(ct ⋅ dc) * dΩ

# Residual component for the diffusive term involving the gradient of μ.
res_ql1(t, μ, dc) = ∫(M*∇(μ)⋅∇(dc)) * dΩ

# Residual component for coupling the chemical potential and the phase field.
res_ql2(t, (c, μ), dμ) = ∫(μ ⋅ dμ - dμ ⋅ dfdc(c) - λ*∇(dμ)⋅∇(c)) * dΩ

# Total quasilinear residual combining the above contributions.
res_ql(t, (c, μ), (dc, dμ)) = res_ql1(t, μ, dc) + res_ql2(t, (c, μ), dμ)

# Alternative formulation for the nonlinear residual:
# res1 corresponds to the time derivative and diffusive contributions.
res1(t, (c, μ), (dc, dμ)) = ∫(∂t(c) ⋅ dc + M*∇(μ)⋅∇(dc)) * dΩ
# res2 corresponds to the chemical potential relation.
res2(t, (c, μ), (dc, dμ)) = ∫(μ ⋅ dμ - dfdc(c) ⋅ dμ - λ*∇(dμ)⋅∇(c)) * dΩ
# The overall residual for the system.
res(t, (c, μ), (dc, dμ)) = res1(t, (c, μ), (dc, dμ)) + res2(t, (c, μ), (dc, dμ))

#------------------------------------------------------------------------------
# Finite Element Operators
#
# These operators are constructed with Automatic Differentiation (AD) to enable
# efficient evaluation of the Jacobian in nonlinear problems.
#

# Nonlinear operator for the full nonlinear formulation.
tfeop_nl = TransientFEOperator(res, X, Y)

# Quasilinear operator using the mass term and quasilinear residual.
tfeop_ql = TransientQuasilinearFEOperator(mass, res_ql, X, Y)

# Semilinear operator using the semilinear mass term.
tfeop_sl = TransientSemilinearFEOperator(mass_sl, res_ql, X, Y)

#------------------------------------------------------------------------------
# Solver Setup
#
# Define a linear solver (LU decomposition) and a nonlinear solver (Newton’s method),
# then encapsulate them within a time-stepping scheme (Theta method).
#
lin_solver = LUSolver()
nl_solver = NLSolver(lin_solver, method=:newton, iterations=50, show_trace=true, ftol=1e-6)

# Time stepping parameters:
Δt = 5 / 1000000      # Time step size.
θ = 0.5               # Theta parameter; θ = 0.5 leads to a Crank–Nicolson scheme.

# Define the Theta method solver which integrates the problem in time.
solver = ThetaMethod(nl_solver, Δt, θ)

# Set initial and final times for the simulation.
t0, tF = 0.0, 50*Δt

#------------------------------------------------------------------------------
# Initial Conditions
#
# The initial phase field is perturbed slightly around 0.63. The corresponding 
# chemical potential is derived from the free energy.
#
V0 = V(t0)  # Get the finite element space at t = 0.

# Define the initial phase field function with a small random perturbation.
function c0(x)
    0.63 + 0.02 * (0.5 - rand())
end

# Define the initial chemical potential μ₀ based on the free energy and the Laplacian.
function μ0(x)
    f(c0(x)) - λ * laplacian(c0)(x)
end

# Create the initial multi-field function (combining c and μ) at t = 0.
X0 = X(t0)
xh0 = interpolate_everywhere([c0, μ0], X0)

#------------------------------------------------------------------------------
# Time Integration: Solve the Transient Problem
#
# The Theta method solver advances the solution from the initial condition xh0
# over the time interval [t0, tF]. The computed solutions for each time step are 
# stored in Xh.
#
Xh = solve(solver, tfeop_nl, t0, tF, xh0)

#------------------------------------------------------------------------------
# Utility Function to Manage VTK Output Files
#
# This function deletes files with a specified extension from a given directory.
#
function delete_vtk_files(folder_path::String, extension::String)
    # Check if the folder exists; raise an error if it does not.
    if !isdir(folder_path)
        error("Folder '$folder_path' does not exist")
    end

    # List all files in the folder.
    files = readdir(folder_path)

    # Loop over files and remove those with the specified extension.
    for file in files
        if endswith(file, extension) # e.g., ".vtu" files.
            full_path = joinpath(folder_path, file)
            rm(full_path)
            println("Deleted: $full_path")
        end
    end
end

#------------------------------------------------------------------------------
# Prepare Output Directories
#
# Check if the temporary directory for VTK output exists. If not, create it.
# If it exists, remove old VTK files to ensure a clean output.
#
if !isdir("tmp/tmp_CahnHilliard")
    mkdir("tmp/tmp_CahnHilliard")
else
    delete_vtk_files("tmp/tmp_CahnHilliard", ".vtu")
end

#------------------------------------------------------------------------------
# Save Simulation Results to VTK Files
#
# Using createpvd, the simulation results are saved for visualization.
# The initial condition and each time-step solution are written as VTK files
# in the designated temporary directory.
#
@time createpvd("tmp/results_CahnHilliard") do pvd
    # Save the initial conditions.
    pvd[0] = createvtk(Ω, "tmp/tmp_CahnHilliard/results_0.vtu", cellfields=["c" => xh0[1], "μ" => xh0[2]])
    # Loop over each time step in the solution and save the corresponding fields.
    for (tn, (cn, μn)) in Xh
        @show tn  # Print the current time step for monitoring progress.
        pvd[tn] = createvtk(Ω, "tmp/tmp_CahnHilliard/results_$tn.vtu", cellfields=["c" => cn, "μ" => μn])
    end
end
