# Linear Thermo-Elasticity Example with Gridap.jl
# ----------------------------------------------
# This script solves a coupled thermal and linear elasticity
# problem on a rectangular domain using Gridap.jl.

using Gridap                    # Core Gridap functionality
import Gridap.Fields: ε         # Strain operator ε(u)
using Gridap.TensorValues

#------------------------------
# 1. Domain and Mesh Definition
#------------------------------
# Define the rectangular domain: (x_min, x_max, y_min, y_max)
# domain    = (0., 5.0, 0., 0.3)
# partition = (100, 10)             # Number of cells in x and y
# model     = CartesianDiscreteModel(domain, partition)

using GridapGmsh
# model = GmshDiscreteModel(joinpath(@__DIR__, "mesh_TransientThermoElsticity.msh"))
include("make_gmsh_model.jl")

#------------------------------
# 2. Boundary Tagging
#------------------------------
# Add custom tags to face_labeling by reusing existing tags:
# "lateral_sides" = union of tags [1,3,7,2,8,4]
# "bottom"        = union of tags [1,5,2]
# "top"           = union of tags [3,6,4]
# add_tag_from_tags!(model.face_labeling, "lateral_sides", [1,3,7,2,8,4])
# add_tag_from_tags!(model.face_labeling, "bottom",         [1,5,2])
# add_tag_from_tags!(model.face_labeling, "top",            [3,6,4])

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

const DT_hole = 10.0

const I2 = SymTensorValue{2,Float64}(1.0,0.0,1.0)

# Thermoelastic stress: σ(ε, dT) = λ·tr(ε)I + 2μ ε − α(3λ+2μ)dT I
σ(ε_in, T_in) = λ*tr(ε_in)*I2 + 2*μ*ε_in - κ*(T_in)*I2
# Pure thermal stress for Neumann-like term: σT(dT) = −α(3λ+2μ)dT I
σT(T_in)  = -κ*(T_in)*I2

#------------------------------
# 4. Finite Element Spaces
#------------------------------
# Displacement: vector field, Dirichlet on lateral_sides
order     = 1
reffe_u   = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V0_u  = TestFESpace( model, reffe_u, dirichlet_tags = ["BottomEdge", "LeftEdge"], 
        dirichlet_masks=[(false, true), (true, false)])

# Temperature: scalar field, Dirichlet on bottom, top, and lateral_sides
reffe_T   = ReferenceFE(lagrangian, Float64, order)
V0_T  = TestFESpace( model, reffe_T, dirichlet_tags = ["circArc"] )
V0 = MultiFieldFESpace([V0_u, V0_T])
#------------------------------
# 5. Integration Measure
#------------------------------
Ω       = Triangulation(model)   # domain triangulation
degree  = 1                       # quadrature degree
dΩ      = Measure(Ω, degree)     # integration measure over Ω


function stepDispTemp(uh_in, Th_in, dt, time, maxSimTime)
    u_bc1(x) = VectorValue(0., 0.)
    u_bc2(x) = VectorValue(0., 0.)
    U_u = TrialFESpace(V0_u, [u_bc1, u_bc2])
    # T_bc1(x) = 50.0*(time-10.0)/maxSimTime
    T_bc1(x) = DT_hole
    U_T = TrialFESpace(V0_T, [T_bc1])
    U = MultiFieldFESpace([U_u, U_T])
    res((u,T), (v, w)) = ∫( (ε(v) ⊙ (σ∘(ε(u), T))) + w*(cV*(T-Th_in)/dt + κ*T0*tr(ε(u-uh_in))/dt) + k*∇(T)⋅∇(w) )dΩ
    op = FEOperator(res, U, V0)
    uh_out, Th_out = Gridap.solve(op)
    return uh_out, Th_out
end

simTimes = 10 .^ range(1, 4, 101)
maxSimTime = maximum(simTimes)
Δts = [simTimes[i+1] - simTimes[i] for i in 1:length(simTimes)-1]

uh = zero(V0_u)
Th = zero(V0_T)

uh, Th = stepDispTemp(uh, Th, Δts[1], simTimes[1], maxSimTime )

#------------------------------
# 8. Post-processing: VTK output
#------------------------------
outdir = "Multiphysics/Thermoelasticity/tmp"
if !isdir(outdir)
    mkdir(outdir)               # create directory if missing
end
# # Write displacement u, temperature T, and total stress σ(ε(u),Th)
writevtk(Ω, joinpath(outdir, "TransientThermoelasticity_10"),
         cellfields = ["u" => uh,
                       "T" => Th,
                       "sigma" => σ∘(ε(uh), Th)])


createpvd(joinpath(outdir, "TransientThermoelasticity")) do pvd               
    for i in 1:length(simTimes)-1
        simTime = simTimes[i+1]
        roundedSimTime = Int(round(simTime))
        @show i, simTime
        global uh, Th = stepDispTemp(uh, Th, Δts[i], simTime, maxSimTime)
        if mod(i, 10)==0
            pvd[i] = createvtk(Ω, joinpath(outdir, "TransientThermoelasticity_$roundedSimTime.vtu"),
                                cellfields = ["u" => uh,
                                              "T" => Th,
                                              "sigma" => σ∘(ε(uh), Th)])
        end
    end 
end