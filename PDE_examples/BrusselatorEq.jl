# Import necessary packages for finite element analysis and ODEs.
using Gridap
using Gridap.GridapODEs

# Define the computational domain in 2D.
# The domain is a rectangle [0,1]×[0,1].
dimain = (0, 1, 0, 1)
# Partition the domain into 32×32 cells for the mesh.
partition = (32, 32)
# Create a Cartesian discrete model (mesh) on the specified domain.
# The 'isperiodic=(true, true)' makes the domain periodic in both spatial directions.
model = CartesianDiscreteModel(dimain, partition, isperiodic=(true, true))

# Define the order of the finite elements (degree 1 Lagrangian elements).
order = 1
# Create a reference finite element (ReferenceFE) for lagrangian type,
# using Float64 as the data type and the specified order.
reffe = ReferenceFE(lagrangian, Float64, order)
# Define a finite element space U_ on the mesh, with conformity to H¹.
U_ = FESpace(model, reffe, conformity=:H1)
# Convert U_ to a transient trial space for time-dependent problems.
U = TransientTrialFESpace(U_)
# Define another finite element space V_ for a second field, with H¹ conformity.
V_ = FESpace(model, reffe, conformity=:H1)
# Similarly convert V_ to a transient trial space.
V = TransientTrialFESpace(V_)

# Create a multi-field transient space X which couples the two fields (u and v).
X = TransientMultiFieldFESpace([U, V])
# Define the corresponding multi-field test space Y for the weak formulation.
Y = MultiFieldFESpace([U_, V_])

# Establish the triangulation of the domain, which represents the mesh cells.
Ω = Triangulation(model)
# Define a measure on the triangulation (for integration),
# here using quadrature of degree 2.
degree = 2
dΩ = Measure(Ω, degree)

# Define the diffusion coefficient (α) used in the problem.
α = 10.0

# Define a spatially and temporally dependent forcing function for the Brusselator.
# The function returns 5.0 inside a circle centered at (0.3, 0.6) with radius 0.1,
# but only active after time t >= 1.1.
brusselator_f(t, x) = (((x[1] - 0.3)^2 + (x[2] - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
# The following creates a time-dependent function f_t that can be evaluated at a specific time.
f_t = (t) -> (x) -> brusselator_f(t, x)

# Define reaction functions for the Brusselator model.
# g1 corresponds to the reaction term for u,
# and includes a nonlinear term u² * v.
g1(u, v) = 1.0 + u*u*v - 4.4*u
# g2 corresponds to the reaction term for v.
g2(u, v) = 3.4*u - u*u*v 

# Define the initial condition for u.
# Uses a profile which is zero at the boundaries since x[2]*(1-x[2]) is zero at 0 and 1,
# raised to the power of 3/2 and scaled by 22.0.
u0(x) = 22.0 * (x[2]*(1 - x[2]))^(3/2)
# Define the initial condition for v in a similar manner,
# with the spatial dependence in the first coordinate.
v0(x) = 27.0 * (x[1]*(1 - x[1]))^(3/2)

# Define the first residual of the weak formulation:
# It includes the time derivative term ∂t(u) (times the test function u_),
# the reaction term g1(u,v), the diffusion term with coefficient α,
# and the forcing term f_t acting on u_.
res1(t, (u, v), (u_, v_)) = ∫(∂t(u)*u_ - g1(u, v)*u_ + α*∇(u) ⋅ ∇(u_) - f_t(t)*u_) * dΩ

# Define the second residual corresponding to the field v:
# It also includes the time derivative, reaction term g2(u,v), and diffusion term.
res2(t, (u, v), (u_, v_)) = ∫(∂t(v)*v_ - g2(u, v)*v_ + α*∇(v) ⋅ ∇(v_)) * dΩ

# Combine the two residuals to form the complete weak form residual for the system.
res(t, (u, v), (u_, v_)) = res1(t, (u, v), (u_, v_)) + res2(t, (u, v), (u_, v_))

# Create the transient finite element operator for the nonlinear problem,
# using the combined residual 'res' along with the trial and test spaces X and Y.
tfeop_nl = TransientFEOperator(res, X, Y)

# Define the time stepping parameters:
Δt = 0.5      # Time step size.
θ = 0.5       # Theta parameter for the Theta method (0.5 corresponds to the Crank-Nicolson scheme).

# Set up solvers:
# Create a linear solver using LU factorization.
lin_solver = LUSolver()
# Configure a nonlinear solver that uses the Newton method with the linear solver,
# set to a maximum of 50 iterations, with trace output enabled for monitoring,
# and a tolerance (ftol) of 1e-6 for convergence.
nl_solver = NLSolver(lin_solver, method=:newton, iterations=50, show_trace=true, ftol=1e-6)

# Initialize the time integration solver with the Theta method,
# combining the nonlinear solver with the chosen time step Δt and theta parameter.
solver = ThetaMethod(nl_solver, Δt, θ)
# Set the starting and final simulation times.
t0, tF = 0.0, 11.5

# Initialize the finite element space at initial time t0.
X0 = X(t0)
# Interpolate the initial conditions (u0 and v0) into the finite element space.
xh0 = interpolate_everywhere([u0, v0], X0)

# Solve the time-dependent nonlinear problem over the interval [t0, tF]
# using the previously defined solver and the initial state xh0.
Xh = solve(solver, tfeop_nl, t0, tF, xh0)

# Check if the directory for saving results exists; if not, create it.
if !isdir("tmp/tmp_Brusselator")
    mkdir("tmp/tmp_Brusselator")
end

# Save the simulation results in VTK format to visualize them later.
# Measure the time taken for the whole process with the '@time' macro.
@time createpvd("tmp/tmp_Brusselator") do pvd
    # Save the initial condition results as a VTK file.
    pvd[0] = createvtk(Ω, "tmp/tmp_Brusselator/results_0.vtu", cellfields=["u" => xh0[1], "v" => xh0[2]])
    # Loop over each time step (tn) and its corresponding solution fields (un and vn).
    for (tn, (un, vn)) in Xh
        @show tn  # Display the current time step to monitor progress.
        # Save the results at the current time step into a VTK file.
        pvd[tn] = createvtk(Ω, "tmp/tmp_Brusselator/results_$tn.vtu", cellfields=["u" => un, "v" => vn])
    end
end
