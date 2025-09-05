# Linear Thermo-Elasticity Example with Gridap.jl
# ----------------------------------------------
# This script solves a coupled thermal and linear elasticity
# problem on a rectangular domain using Gridap.jl.

using Gridap                    # Core Gridap functionality
import Gridap.Fields: ε         # Strain operator ε(u)

#------------------------------
# 1. Domain and Mesh Definition
#------------------------------
# Define the rectangular domain: (x_min, x_max, y_min, y_max)
domain    = (0., 5.0, 0., 0.3)
partition = (100, 10)             # Number of cells in x and y
model     = CartesianDiscreteModel(domain, partition)

#------------------------------
# 2. Boundary Tagging
#------------------------------
# Add custom tags to face_labeling by reusing existing tags:
# "lateral_sides" = union of tags [1,3,7,2,8,4]
# "bottom"        = union of tags [1,5,2]
# "top"           = union of tags [3,6,4]
add_tag_from_tags!(model.face_labeling, "lateral_sides", [1,3,7,2,8,4])
add_tag_from_tags!(model.face_labeling, "bottom",         [1,5,2])
add_tag_from_tags!(model.face_labeling, "top",            [3,6,4])

#------------------------------
# 3. Material and Thermal Parameters
#------------------------------
const T0 = 293.0                # Initial temperature
const E  = 70.0e3               # Young's modulus
const ν  = 0.3                  # Poisson's ratio
const μ  = E/(2*(1+ν))          # Shear modulus (Lame μ)
const λ  = E*ν/((1+ν)*(1-2*ν))  # First Lame parameter λ
const α  = 2.31e-5              # Thermal expansion coefficient
const κ  = α*(3*λ + 2*μ)
const ρ  = 2700.0               # density        
const cV = 910e-6 * ρ           # specific heat per unit volumr at constant strain
const k  = 237e-6               # Thermal conductivity

# Thermoelastic stress: σ(ε, dT) = λ·tr(ε)I + 2μ ε − α(3λ+2μ)dT I
σ(ϵ, dT) = λ*tr(ϵ)*one(ϵ) + 2*μ*ϵ - κ*dT*one(ϵ)
# Pure thermal stress for Neumann-like term: σT(dT) = −α(3λ+2μ)dT I
σT(dT)  = -κ*dT*one(TensorValue{2,2,Float64})

#------------------------------
# 4. Finite Element Spaces
#------------------------------
# Displacement: vector field, Dirichlet on lateral_sides
order     = 1
reffe_u   = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
testFE_u  = TestFESpace( model, reffe_u, dirichlet_tags = "lateral_sides" )
trialFE_u = TrialFESpace(testFE_u, VectorValue(0.0,0.0))  # zero displacement bc

# Temperature: scalar field, Dirichlet on bottom, top, and lateral_sides
reffe_T   = ReferenceFE(lagrangian, Float64, order)
testFE_T  = TestFESpace( model, reffe_T,
                         dirichlet_tags = ["bottom","top","lateral_sides"] )
trialFE_T = TransientTrialFESpace(testFE_T, [(t)->50.0, (t)->0.0, (t)->0.0])        # prescribed T on tags

X = TransientMultiFieldFESpace([trialFE_u, trialFE_T])
Y = MultiFieldFESpace([testFE_u, testFE_T])

#------------------------------
# 5. Integration Measure
#------------------------------
Ω       = Triangulation(model)   # domain triangulation
degree  = 2                       # quadrature degree
dΩ      = Measure(Ω, degree)     # integration measure over Ω

# #------------------------------
# # 6. Thermal Subproblem
# #------------------------------
# # Bilinear form a_T(T, Tt) = ∫ ∇T ⋅ ∇Tt dΩ
# a_T(T, Tt) = ∫( ∇(T) ⋅ ∇(Tt) ) * dΩ
# # Linear form b_T(Tt) = ∫ 0 · Tt dΩ (no volumetric heat source)
# b_T(Tt)   = ∫( 0.0 * Tt ) * dΩ

# # Assemble and solve for temperature Th
# op_T = AffineFEOperator(a_T, b_T, trialFE_T, testFE_T)
# Th   = solve(op_T)

# #------------------------------
# # 7. Elasticity Subproblem
# #------------------------------
# # Bilinear form a_u(u, v) = ∫ σ(ε(u), Th) : ε(v) dΩ
# a_u(u, v) = ∫( σ∘(ε(u), Th) ⊙ ε(v) ) * dΩ
# # Linear form b_u(v) = ∫ σT(Th) : ε(v) dΩ (thermal load)
# b_u(v)   = ∫( σT∘(Th) ⊙ ε(v) ) * dΩ

# # Assemble and solve for displacement uh
# op_u = AffineFEOperator(a_u, b_u, trialFE_u, testFE_u)
# uh   = solve(op_u)

# res_mech((u, T), (v, Tt)) = ∫(σ∘(ε(u), T-T0) ⊙ ε(v)) * dΩ
# res_thermal(t, (u, T), (v, Tt)) =  ∫( ∇(T) ⋅ ∇(Tt) ) * dΩ

# # As a TransientFEOperator
# res(t, (u, T), (v, Tt)) = res_mech((u, T), (v, Tt)) + res_thermal(t, (u, T), (v, Tt))

# # Nonlinear operator for the full nonlinear formulation.
# tfeop_nl = TransientFEOperator(res, X, Y)


#------------------------------
# 8. Post-processing: VTK output
#------------------------------
outdir = "Multiphysics/Thermoelasticity/tmp"
if !isdir(outdir)
    mkdir(outdir)               # create directory if missing
end
# # Write displacement u, temperature T, and total stress σ(ε(u),Th)
# writevtk(Ω, joinpath(outdir, "LinearThermalelasticity"),
#          cellfields = ["u" => uh,
#                        "T" => Th,
#                        "sigma" => σ∘(ε(uh), Th)])

timeSteps = 10 .^ range(1, 4, 101)
Δts = [timeSteps[i+1] - timeSteps[i] for i in 1:length(timeSteps)-1]

for time in timeSteps
    
end 
