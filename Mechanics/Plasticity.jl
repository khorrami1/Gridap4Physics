using Gridap
using Gridap.TensorValues
import ForwardDiff as FD
using StaticArrays

T = Float64
model = CartesianDiscreteModel((T(0),T(1),T(0),T(0.002),T(0),T(0.1)), (100, 10, 1))

Dim = length(size(model.grid.node_coords))

function KronckerDelta(i, j)
    if i==j
        return 1
    else
        return 0
    end
end

E = T(69e3)
ν = T(0.3)              
# Lame parameters for linear elasticity model
λ = (E*ν)/((1+ν)*(1-2*ν))  # First Lame parameter
μ = E/(2*(1+ν))            # Shear modulus (second Lame parameter)

Celas = zero(SymFourthOrderTensorValue{Dim, T})

Celas_data = zeros(T, ((Dim*(Dim+1)) ÷ 2)^2)


function sym_index(i, j, D)
    @assert 1 ≤ i ≤ j ≤ D
    return div((2D - i + 1)*(i - 1), 2) + (j - i + 1)
end

function sym4_index(i,j,k,l,D)
    @assert 1 ≤ i ≤ j ≤ D
    @assert 1 ≤ k ≤ l ≤ D
    I = sym_index(i,j,D)
    J = sym_index(k,l,D)
    N = (D*(D+1)) ÷ 2
    return (I - 1) * N + J
end

for i=1:Dim
    for j=i:Dim
        for k=1:Dim
            for l=k:Dim
                Celas_data[sym4_index(i,j,k,l,Dim)] = λ*KronckerDelta(i,j)*KronckerDelta(k,l) + 
                    μ*(KronckerDelta(i,k)*KronckerDelta(j,l) + KronckerDelta(i,l)*KronckerDelta(j,k))
            end
        end
    end
end

Celas = SymFourthOrderTensorValue(Celas_data...)

yieldStress(ϵ) = T(376.9*(0.0059+ϵ)^0.152)

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    M::T
    N::T
    L::T
end 

function (f::Yield_Hill48)(S1, S2, S3, S4, S5, S6)
    return sqrt( (f.F*S1*S1 + f.G*S4*S4 + f.H*S6*S6) / 
        (f.F*f.G + f.F*f.H + f.G*f.H) + 2*S5*S5/f.L + 2*S3*S3/f.M + 2*S2*S2/f.N )
end

R0 =  T(0.84)
R45 = T(0.64)
R90 = T(1.51)

F = R0/(R90*(R0+1))
G = 1/(1+R0)
H = R0/(1+R0)

N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
L = N # must be checked!
M = N # must be checked!

yield_Hill48 = Yield_Hill48(F, G, H, M, N, L)
# yieldFunction1(S::SymTensorValue{Dim,T}) = yield_Hill48(S)
yieldFunction1(S1, S2, S3, S4, S5, S6) = yield_Hill48(S1, S2, S3, S4, S5, S6)


struct GeneralPlastic{D,T}
    Celas :: SymFourthOrderTensorValue{D,T}
    yieldStress :: Function
    yieldFunc :: Function
end

struct GeneralPlasticState{D,T}
    εp :: SymTensorValue{D,T}
    εe :: SymTensorValue{D,T}
    σ  :: SymTensorValue{D,T}
    λp  :: T
end

initial_material_state(::GeneralPlastic) = GeneralPlasticState(zero(SymTensorValue{Dim,T}),
    zero(SymTensorValue{Dim,T}), zero(SymTensorValue{Dim,T}), T(0.0))

material = GeneralPlastic(Celas, yieldStress, yieldFunction1)

initial_material_state(material)


σ(ε) = Celas ⊙ ε


order = 1

reffe = ReferenceFE(lagrangian, VectorValue{Dim, T}, order)
test_FE = TestFESpace(model, reffe; conformity=:H1,
    dirichlet_tags = ["tag_7", "tag_1", "tag_3"])

g0(x) = VectorValue(zeros(T, Dim))

trial_FE = TrialFESpace(test_FE, [g0, g0, g0])

degree = 2*order 
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

εp_state1 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
εe_state1 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
σ_state1 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
λ_state1 = CellState(T(0.0), dΩ)

εp_state2 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
εe_state2 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
σ_state2 = CellState(zero(SymTensorValue{Dim,T}), dΩ)
λ_state2 = CellState(T(0.0), dΩ)


# Body force (e.g., gravity) acting downward in y-direction
forceValues = zeros(T, Dim)
forceValues[2] = -1e-3
f = VectorValue(forceValues...)

nls = NLSolver(show_trace=true, method=:newtom)
solver = FESolver(nls)

nSteps = 10
cache = nothing

SS = one(SymTensorValue{3, Float64})

@inline function grad_wrt_entries(f::Function, S::T) where {T<:SymTensorValue}
    return T(FD.derivative(S1->f(S1, S[2], S[3], S[4], S[5], S[6]), S[1]),
             FD.derivative(S2->f(S[1], S2, S[3], S[4], S[5], S[6]), S[2]),
             FD.derivative(S3->f(S[1], S[2], S3, S[4], S[5], S[6]), S[3]),
             FD.derivative(S4->f(S[1], S[2], S[3], S4, S[5], S[6]), S[4]),
             FD.derivative(S5->f(S[1], S[2], S[3], S[4], S5, S[6]), S[5]),
             FD.derivative(S6->f(S[1], S[2], S[3], S[4], S[5], S6), S[6]))
end


# This method is the lowest allocation
@time out = grad_wrt_entries(yieldFunction1, SS)

# This approch of differentiating is less efficient
# cfg = FD.GradientConfig(x->yieldFunction1(x...), SVector{6, T}(SS.data), FD.Chunk{6}())
# @time FD.gradient(x->yieldFunction1(x...), SVector{6, T}(SS.data), cfg)

function σ!(ε)

end

function step(uh_in, step, cache)
    b = step/nSteps * forceValues 
    res(u,v) = ∫( ε(v) ⊙ (σ! ∘ (ε(u), state_old, state_new)) - v ⋅ b)*dΩ
    op = FEOperator(res, trial_FE, test_FE)
    uh_out, cache = solve!(uh_in, solver, op, cache)
    update_state!(new_state, )
    return uh_out, cache
end

for step in 1:nSteps

    println("\n+++ Solving ...\n")

    uh, cache = step(uh_in, step, cache)

    

end

#---------------------------------
# 5. Post-Processing / Visualization
#---------------------------------
# Ensure output directory exists
if !isdir("Mechanics/tmp")
    mkdir("Mechanics/tmp")
end
# Write VTK files: displacement uh and computed stress σ∘ε(uh)
writevtk(Ω, "Mechanics/tmp/plasticity", 
         cellfields = ["uh" => uh, "sigma" => σ∘ε(uh)])

writevtk(model, "Mechanics/tmp/model_plasticity")


# ============================================
using Gridap
using Gridap.TensorValues
import ForwardDiff as FD
using StaticArrays
using LinearAlgebra

# ------------------------------------------------------------
# 0) Mesh & basic types
# ------------------------------------------------------------
const T = Float64
model = CartesianDiscreteModel((T(0),T(0.1), T(0),T(0.1), T(0),T(0.001)), (100,100,1))
const Dim = 3  # 3D block

# ------------------------------------------------------------
# 1) Elasticity constants (your values)
# ------------------------------------------------------------
E = T(69e3)
ν = T(0.3)
λ = (E*ν)/((1+ν)*(1-2*ν))   # First Lamé
μ = E/(2*(1+ν))             # Shear modulus

# Build 4th-order elasticity tensor C with minor symmetries in the symmetric storage you set up
KroneckerDelta(i,j) = i==j ? one(T) : zero(T)

function sym_index(i, j, D)
  @assert 1 ≤ i ≤ j ≤ D
  return div((2D - i + 1)*(i - 1), 2) + (j - i + 1)
end
function sym4_index(i,j,k,l,D)
  @assert 1 ≤ i ≤ j ≤ D
  @assert 1 ≤ k ≤ l ≤ D
  I = sym_index(i,j,D)
  J = sym_index(k,l,D)
  N = (D*(D+1)) ÷ 2
  return (I - 1) * N + J
end

Celas_data = zeros(T, ((Dim*(Dim+1)) ÷ 2)^2)
for i=1:Dim, j=i:Dim, k=1:Dim, l=k:Dim
  Celas_data[sym4_index(i,j,k,l,Dim)] =
    λ*KroneckerDelta(i,j)*KroneckerDelta(k,l) +
    μ*(KroneckerDelta(i,k)*KroneckerDelta(j,l) + KroneckerDelta(i,l)*KroneckerDelta(j,k))
end
const Celas = SymFourthOrderTensorValue{Dim,T}(Celas_data...)

# Convenience: elastic predictor
σe(ε::SymTensorValue{Dim,T}) = Celas ⊙ ε

# ------------------------------------------------------------
# 2) Hill‑48 equivalent stress (as provided) + hardening
# ------------------------------------------------------------
yieldStress(εpbar) = T(376.9*(0.0059 + εpbar)^0.152)
# derivative for Newton
dyieldStress(εpbar) = T(376.9 * 0.152 * (0.0059 + εpbar)^(0.152 - 1))

struct Yield_Hill48{T}
  F::T; G::T; H::T; M::T; N::T; L::T
end

function (f::Yield_Hill48)(S1,S2,S3,S4,S5,S6)
  # NOTE: this is exactly your formula. (Classic Hill48 uses stress differences,
  # but here we keep your definition to stay faithful to your setup.)
  return sqrt( (f.F*S1*S1 + f.G*S4*S4 + f.H*S6*S6)/(f.F*f.G + f.F*f.H + f.G*f.H)
               + 2*S5*S5/f.L + 2*S3*S3/f.M + 2*S2*S2/f.N )
end

# R-values and coefficients (kept as you wrote)
R0  =  T(0.84)
R45 =  T(0.64)
R90 =  T(1.51)
F = R0/(R90*(R0+1))
G = 1/(1+R0)
H = R0/(1+R0)
N_hill = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
L_hill = N_hill
M_hill = N_hill
const yield_Hill48 = Yield_Hill48(F,G,H,M_hill,N_hill,L_hill)

function yieldFunction1(S1,S2,S3,S4,S5,S6)
  return yield_Hill48(S1,S2,S3,S4,S5,S6)
end

# Wrapper: eqv stress and its gradient wrt the 6 symmetric entries (via ForwardDiff)
hill_eqv(S::SymTensorValue{Dim,T}) = yield_Hill48(S[1],S[2],S[3],S[4],S[5],S[6])

@inline function grad_wrt_entries(f::Function, S::SymTensorValue{Dim,T})
  return SymTensorValue{Dim,T}(
    FD.derivative(S1->f(S1, S[2], S[3], S[4], S[5], S[6]), S[1]),
    FD.derivative(S2->f(S[1], S2, S[3], S[4], S[5], S[6]), S[2]),
    FD.derivative(S3->f(S[1], S[2], S3, S[4], S[5], S[6]), S[3]),
    FD.derivative(S4->f(S[1], S[2], S[3], S4, S[5], S[6]), S[4]),
    FD.derivative(S5->f(S[1], S[2], S[3], S[4], S5, S[6]), S[5]),
    FD.derivative(S6->f(S[1], S[2], S[3], S[4], S[5], S6), S[6]),
  )
end
# grad_hill(σ::SymTensorValue{Dim,T}) = grad_wrt_entries(hill_eqv, σ)

# Helpers
frob_norm2(S::SymTensorValue{Dim,T}) = S ⊙ S
frob_norm(S::SymTensorValue{Dim,T}) = sqrt(frob_norm2(S))

# ------------------------------------------------------------
# 3) Stateful constitutive update (return mapping at Gauss points)
# ------------------------------------------------------------
# Internal variables stored as CellStates: plastic strain tensor εp and scalar ε̄p (equiv. plastic strain)
# We also keep the cumulative plastic multiplier λp, mostly for inspection/output.
#
# Interface new_state(εp_in, εpbar_in, λp_in, ε_in) -> (is_plastic::Bool, εp_out, εpbar_out, λp_out)
# as required by Gridap's update_state! (first value Bool, rest are the new states).  See tutorial pattern.  :contentReference[oaicite:1]{index=1}
#
function new_state(εp_in::SymTensorValue{Dim,T}, εpbar_in::T, λp_in::T, ε_in::SymTensorValue{Dim,T})
  # Elastic predictor
  εe_tr = ε_in - εp_in
  σ_tr  = σe(εe_tr)
  σeq_tr = hill_eqv(σ_tr)
  σy_n   = yieldStress(εpbar_in)

  # Elastic step?
  if σeq_tr ≤ σy_n + T(1e-10)
    return (false, εp_in, εpbar_in, λp_in)
  end

  # Plastic: implicit BE with iterative Δγ update (direction recomputed each iter)
  γ = zero(T)
  σ = σ_tr
  maxit = 30
  tol = T(1e-10)

  for it = 1:maxit
    # n = grad_hill(σ)                # ∂f/∂σ (same units as 1/Pa here)
    n = grad_wrt_entries(yieldFunction1, σ)
    Cn = Celas ⊙ n
    nCn = n ⊙ Cn                    # n:C:n  (scalar)
    rfac = sqrt(T(2)/T(3)) * frob_norm(n)  # Δε̄p ≈ rfac*Δγ (associated flow, Frobenius norm)
    σeq = hill_eqv(σ)
    σy  = yieldStress(εpbar_in + γ*rfac)
    res = σeq - σy

    if abs(res) ≤ tol
      break
    end

    # Approximate derivative d(res)/dγ (neglecting dependence of n on γ)
    H = dyieldStress(εpbar_in + γ*rfac)
    dres = -(nCn) - H*rfac
    # Safeguards
    if abs(dres) < T(1e-20)
      dres = sign(dres + T(1e-30)) * T(1e-20)
    end

    γ_new = γ - res/dres
    γ = max(γ_new, zero(T))         # keep non-negative
    # Update stress with current direction (standard corrector)
    σ = σ_tr - γ * Cn
  end

  # Update internal variables
  n = grad_wrt_entries(yieldFunction1, σ)
  Δεp = γ * n
  rfac = sqrt(T(2)/T(3)) * frob_norm(n)
  εp_out   = εp_in + Δεp
  εpbar_out = εpbar_in + rfac*γ
  λp_out    = λp_in + γ

  return (true, εp_out, εpbar_out, λp_out)
end

# Stress as function of current strain and *current states* (states are updated on the fly but not stored yet)
function σ(ε_in::SymTensorValue{Dim,T}, εp_state::SymTensorValue{Dim,T}, εpbar_state::T, λp_state::T)
  _, εp_out, _, _ = new_state(εp_state, εpbar_state, λp_state, ε_in)
  return σe(ε_in - εp_out)
end

# Linearization: dσ = Cep : dε.  We return Cep*dε directly to avoid building 4th-order dyads explicitly.
function dσ(dε_in::SymTensorValue{Dim,T}, ε_in::SymTensorValue{Dim,T}, state)
  plastic, εp_out, εpbar_out, _ = state
  if !plastic
    return σe(dε_in)
  end
  # Recompute at updated state
  σ_cur = σe(ε_in - εp_out)
  n = grad_wrt_entries(yieldFunction1, σ_cur)
  Cn = Celas ⊙ n
  nCn = n ⊙ Cn
  rfac = sqrt(T(2)/T(3)) * frob_norm(n)
  H = dyieldStress(εpbar_out)
  denom = nCn + H*rfac
  # Cep : dε = C : dε - (C:n) * (n : C : dε) / denom
  Cdε = Celas ⊙ dε_in
  scalar = (n ⊙ (Celas ⊙ dε_in)) / denom
  return Cdε - scalar * Cn
end

# ------------------------------------------------------------
# 4) FE spaces, measures, states, solver
# ------------------------------------------------------------
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{Dim,T}, order)

# Boundary conditions: clamp faces x=0, y=0, z=0 to avoid rigid modes
labeling = get_face_labeling(model)
test_FE  = TestFESpace(model, reffe; conformity=:H1, labels=labeling,
                       dirichlet_tags = ["tag_01","tag_03","tag_05"])
g0(x) = VectorValue(zero(T), zero(T), zero(T))
trial_FE = TrialFESpace(test_FE, [g0, g0, g0])

degree = 2*order
Ω  = Triangulation(model)
dΩ = Measure(Ω, degree)

# Body force
forceValues = zeros(T, Dim); forceValues[3] = -1.0*100
b = VectorValue(forceValues...)

# Internal variable states at Gauss points
εp_state  = CellState(zero(SymTensorValue{Dim,T}), dΩ)  # plastic strain tensor
εpb_state = CellState(T(0.0), dΩ)                       # equivalent plastic strain ε̄p
λp_state  = CellState(T(0.0), dΩ)                       # cumulative plastic multiplier (optional)

# Nonlinear solver
nls    = NLSolver(show_trace=true, method=:newton)
solver = FESolver(nls)

# ------------------------------------------------------------
# 5) Load stepping
# ------------------------------------------------------------
nSteps = 10

function step(uh_in, factor, cache)
  bf = factor * b
  # Residual and Jacobian (stateful material pattern) — same structure as Gridap tutorial. :contentReference[oaicite:2]{index=2}
  res(u,v) = ∫( ε(v) ⊙ ( σ ∘ (ε(u), εp_state, εpb_state, λp_state) ) - v⋅bf ) * dΩ
  jac(u,du,v) = ∫( ε(v) ⊙ ( dσ ∘ (ε(du), ε(u), new_state ∘ (εp_state, εpb_state, λp_state, ε(u))) ) ) * dΩ
  op = FEOperator(res, jac, trial_FE, test_FE)
  uh_out, cache = solve!(uh_in, solver, op, cache)
  # Commit updated states at converged uh_out
  update_state!(new_state, εp_state, εpb_state, λp_state, ε(uh_out))
  return uh_out, cache
end

uh = zero(trial_FE)
cache = nothing

for istep in 1:nSteps
  factor = istep / nSteps
  @info "+++ Solving load step $istep / $nSteps (factor = $factor) +++"
  uh, cache = step(uh, factor, cache)
end

# ------------------------------------------------------------
# 6) Post-processing
# ------------------------------------------------------------
mkpath("Mechanics/tmp")

# Project scalar ε̄p for visualization (L2 projection from Gauss points to a discontinuous FE space)
function project_cellstate(q, model, dΩ, order)
  reffe = ReferenceFE(lagrangian, T, order)
  V = FESpace(model, reffe, conformity=:L2)
  a(u,v) = ∫( u*v ) * dΩ
  l(v)   = ∫( v*q ) * dΩ
  op = AffineFEOperator(a,l,V,V)
  return solve(op)
end

εpbar_h = project_cellstate(εpb_state, model, dΩ, order)

writevtk(Ω, "Mechanics/tmp/plasticity",
  cellfields = [
    "uh"      => uh,
    "eps"     => ε(uh),
    #"sigma"   => (σ ∘ (ε(uh), εp_state, εpb_state, λp_state)),
    "epbar"   => εpbar_h
  ])

writevtk(model, "Mechanics/tmp/model_plasticity")

