using Gridap
using SparseArrays, LinearAlgebra
using Random

# --- Mesh & function spaces (no Dirichlet tags here) ---
domain    = (0.0, 1.0, 0.0, 1.0)
partition = (100, 100)
model     = CartesianDiscreteModel(domain, partition)

order   = 1
reffe_T = ReferenceFE(lagrangian, Float64, order)

V = TestFESpace(model, reffe_T)  # no boundary conditions
U = TrialFESpace(V)

Ω      = Triangulation(model)
degree = 2
dΩ     = Measure(Ω, degree)

# --- Laplace weak form (no sources) ---
a_T(T,Tt) = ∫( ∇(T)⋅∇(Tt) ) * dΩ
b_T(Tt)   = ∫( 0.0 * Tt    ) * dΩ


# Boundary tags
add_tag_from_tags!(model.face_labeling, "bottom", [1,5,2])
add_tag_from_tags!(model.face_labeling, "top",    [3,6,4])

# Dirichlet problem (this reproduces your original code to get Th_bc)
Vbc = TestFESpace(model, reffe_T; dirichlet_tags = ["bottom","top"])
Ubc = TrialFESpace(Vbc, [50.0, 0.0])

op_bc  = AffineFEOperator(a_T, b_T, Ubc, Vbc)
Th_bc  = solve(op_bc)  # "ground truth" temperature field with top/bottom BCs


op = AffineFEOperator(a_T, b_T, U, V)

matrix = op.op.matrix
rhs = op.op.vector

# --- User-provided interior Dirichlet data on nodal DOFs ---
# IMPORTANT: fixed_dofs are 1-based global DOF indices in the FE space U/V
# Example placeholder (replace with your own):
# fixed_dofs = [100, 2500, 5100]             # e.g., interior nodes you want to pin
# fixed_vals = [37.2, 15.8, 43.5]            # temperature at those nodes

all_dofs = collect(1:length(model.grid.node_coords))

nRandomNodes = 200
randomNodes = shuffle(1:length(model.grid.node_coords))[1:nRandomNodes]

fixed_dofs = randomNodes
free_dofs = setdiff(all_dofs, fixed_dofs)
fixed_vals = Th_bc(model.grid.node_coords[randomNodes]) 

@assert length(fixed_dofs) == length(fixed_vals)

# --- Solve ---
x = zeros(length(all_dofs))
x[fixed_dofs] .= fixed_vals
x[free_dofs] .= matrix[free_dofs, free_dofs] \ (rhs[free_dofs] - matrix[free_dofs, fixed_dofs]*x[fixed_dofs])

# --- Wrap solution as an FE function and write VTK ---
Th = FEFunction(U, x)

outdir = "Multiphysics/Thermoelasticity/tmp"
isdir(outdir) || mkdir(outdir)
writevtk(Ω, joinpath(outdir,"HeatTransfer_interiorDirichlet"),
         cellfields = ["T" => Th, "T_bc" => Th_bc])
