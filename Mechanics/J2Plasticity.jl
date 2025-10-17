
using Gridap
using Gridap.TensorValues

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

E = T(210e3)
ν = T(0.3)
σ0 = T(200.0)
H = T(10e-1)              
# Lame parameters for linear elasticity model
λ = (E*ν)/((1+ν)*(1-2*ν))  # First Lame parameter
μ = E/(2*(1+ν))            # Shear modulus (second Lame parameter)

D = zero(SymFourthOrderTensorValue{Dim, T})

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

D = SymFourthOrderTensorValue(Celas_data...)

struct J2Plasticity{T, S <: SymFourthOrderTensorValue{Dim, T}}
    G::T  # Shear modulus
    K::T  # Bulk modulus
    σ0::T # Initial yield limit
    H::T  # Hardening modulus
    D::S  # Elastic stiffness tensor
end

function J2Plasticity(;E, ν, σ0, H)
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    return J2Plasticity(G, K, σ0, H, Celas)
end

# State variables
struct J2PlasticityState{T, S <: SymTensorValue{Dim, T}}
    ϵp::S # plastic strain
    k::T # hardening variable
end

function initial_material_state(::J2Plasticity{T, S}) where {T, S}
    return J2PlasticityState(zero(SymTensorValue{Dim, T}), 0.0)
end

function material_response(
    material::J2Plasticity, ϵ::SymTensorValue{Dim, T}, state::J2PlasticityState{T, S}, Δt, cache, extras) where {T, S}
    ## unpack some material parameters
    G = material.G
    H = material.H

    ## We use (•)ᵗ to denote *trial*-values
    σᵗ = material.D ⊙ (ϵ - state.ϵp) # trial-stress
    sᵗ = dev(σᵗ)         # deviatoric part of trial-stress
    J₂ = 0.5 * sᵗ ⊙ sᵗ  # second invariant of sᵗ
    σᵗₑ = sqrt(3.0*J₂)   # effective trial-stress (von Mises stress)
    σʸ = material.σ0 + H * state.k # Previous yield limit

    φᵗ  = σᵗₑ - σʸ # Trial-value of the yield surface

    if φᵗ < 0.0 # elastic loading
        return σᵗ, material.D, state
    else # plastic loading
        h = H + 3G
        μ =  φᵗ / h   # plastic multiplier

        c1 = 1 - 3G * μ / σᵗₑ
        s = c1 * sᵗ           # updated deviatoric stress
        σ = s + vol(σᵗ)       # updated stress

        ## Compute algorithmic tangent stiffness ``D = \frac{\Delta \sigma }{\Delta \epsilon}``
        κ = H * (state.k + μ) # drag stress
        σₑ = material.σ0 + κ  # updated yield surface

        δ(i,j) = i == j ? 1.0 : 0.0
        Isymdev(i,j,k,l)  = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)
        Q(i,j,k,l) = Isymdev(i,j,k,l) - 3.0 / (2.0*σₑ^2) * s[i,j]*s[k,l]
        b = (3G*μ/σₑ) / (1.0 + 3G*μ/σₑ)

        Dtemp(i,j,k,l) = -2G*b * Q(i,j,k,l) - 9G^2 / (h*σₑ^2) * s[i,j]*s[k,l]
        D = material.D + SymmetricTensor{4, 3}(Dtemp)

        ## Return new state
        Δϵᵖ = 3/2 * μ / σₑ * s # plastic strain
        ϵp = state.ϵp + Δϵᵖ    # plastic strain
        k = state.k + μ        # hardening variable
        return σ, D, J2PlasticityState(ϵp, k)
    end
end

##############################################################
# Test one element 

using Plots

function uniaxialTest(loadingRange, Δε)
    m = J2Plasticity(;E=210e3, ν=0.3, σ0=200.0, H=10e-1)
    #cache = get_cache(m)
    state = initial_material_state(m)
    e11_all = Float64[]
    s11_all = Float64[]
    push!(e11_all, 0.0)
    push!(s11_all, 0.0)
    ε = zero(SymTensorValue{3,Float64})
    for e11 in loadingRange
        ε += Δε
        σ, ∂σ∂ε, state = material_response(m, ε, state, nothing, nothing, nothing)
        push!(e11_all, e11)
        push!(s11_all, σ[1,1])
    end
    return e11_all, s11_all, state
end

loadingRange = range(0.0, 0.002, 201)
Δε = SymTensorValue{3, Float64}( loadingRange.step.hi, 0.0, 0.0, 0.0, 0.0, 0.0)
e11_all, s11_all, state = uniaxialTest(loadingRange, Δε)
p = plot(e11_all, s11_all)
loadingRange = range(0.0, 0.002, 201)
Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : (i==2 && j==2 ? loadingRange.step.hi : 0.0))
e11_all, s11_all, state = uniaxialTest(loadingRange, Δε)
p = plot(e11_all, s11_all)

# End of one element test
##############################################################