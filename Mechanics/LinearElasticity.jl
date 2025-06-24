# Linear Elasticity Example with Gridap.jl
# ----------------------------------------
# Load the core Gridap packages for mesh generation, geometry, and reference finite elements
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs

# Define material properties
# E: Young's modulus (stiffness)
# ν: Poisson's ratio (lateral contraction)
const E = 1.0e5             # N/m^2 (example value)
const ν = 0.3               # dimensionless
# Lame parameters for linear elasticity model
const λ = (E*ν)/((1+ν)*(1-2*ν))  # First Lame parameter
const μ = E/(2*(1+ν))            # Shear modulus (second Lame parameter)

# Define the stress-strain relationship: σ(ε) = λ tr(ε) I + 2 μ ε
σ(ϵ) = λ*tr(ϵ)*one(ϵ) + 2*μ*ϵ

#------------------------------
# 1. Domain and Discretization
#------------------------------
# Define a rectangular domain: (x_min, x_max, y_min, y_max)
domain = (0.0, 25.0, 0.0, 1.0)
# Specify a uniform grid partition: number of cells in x and y directions
partition = (250, 10)
# Build a Cartesian mesh and wrap as a DiscreteModel
model = CartesianDiscreteModel(domain, partition)

#---------------------------------
# 2. Finite Element Space Setup
#---------------------------------
# Vector g0 specifies homogeneous Dirichlet boundary values (zero displacement)
g0 = VectorValue(0.0, 0.0)
# Triangulation of the domain for integration
Ω = Triangulation(model)
# Integration degree (polynomial exactness) for numerical quadrature
degree = 2
# Create a measure over the cells of Ω for integrating volume forms
dΩ = Measure(Ω, degree)

# Order of Lagrangian finite elements
order = 1
# Define a vector-valued reference FE (Lagrangian) of given order
reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
# Test space V: functions vanishing on Dirichlet boundary tagged "tag_7", "tag_1", and "tag_3"
V = TestFESpace(model, reffe; dirichlet_tags = ["tag_7", "tag_1", "tag_3"])
# Trial space U: same as V but with prescribed boundary data g0
U = TrialFESpace(V, [g0, g0, g0])

#---------------------------------
# 3. Weak Formulation
#---------------------------------
# Body force (e.g., gravity) acting downward in y-direction
f = VectorValue(0.0, -1e-3)
# Define the bilinear form a(u,v) = ∫ σ(ε(u)) : ε(v) dΩ
a(u, v) = ∫(σ∘ε(u) ⊙ ε(v)) * dΩ
# Define the linear form l(v) = ∫ f ⋅ v dΩ\ n
l(v) = ∫(f ⋅ v) * dΩ

#---------------------------------
# 4. Solve the Finite Element Problem
#---------------------------------
# Build an affine FE operator from forms a and l, spaces U and V
op = AffineFEOperator(a, l, U, V)
# Compute the FE solution uh
uh = solve(op)

#---------------------------------
# 5. Post-Processing / Visualization
#---------------------------------
# Ensure output directory exists
if !isdir("Mechanics/tmp")
    mkdir("Mechanics/tmp")
end
# Write VTK files: displacement uh and computed stress σ∘ε(uh)
writevtk(Ω, "Mechanics/tmp/linearelasticity", 
         cellfields = ["uh" => uh, "sigma" => σ∘ε(uh)])

writevtk(model, "Mechanics/tmp/model_linearelasticity")