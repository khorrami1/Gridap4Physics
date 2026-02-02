# Geometrically nonlinear Reissner–Mindlin plate (Total Lagrangian membrane + Newton)
# - Flat rectangular plate mid-surface (10 x 10)
# - Thickness h = 1.0
# - E = 210, ν = 0.3, shear correction κ = 5/6
# - BCs kept as in the provided code:
#     - u, w fixed on "right"
#     - θ prescribed on "left": θ = (0, π/10)
# - Uses reduced integration for shear to mitigate locking

using Gridap
using LinearAlgebra
using StaticArrays

function main()
  # --- Geometry & mesh ---
  nels = (10,10)
  Lx, Ly = 10.0, 10.0
  h  = 1.0
  E  = 210.0
  ν  = 0.3
  κs = 5/6    # shear correction factor

  model = CartesianDiscreteModel((0.0,Lx, 0.0,Ly), nels)
  labels = get_face_labeling(model)
  # In Gridap's Cartesian models
  add_tag_from_tags!(labels,"left", [1, 3, 7])
  add_tag_from_tags!(labels,"right", [2, 4, 8])
  add_tag_from_tags!(labels,"bottom",[1, 2, 5])
  add_tag_from_tags!(labels,"top", [3, 4, 6])

  Ω  = Triangulation(model)
  # Under-integration for shear to reduce locking
  dΩm = Measure(Ω, 2)  # membrane
  dΩb = Measure(Ω, 2)  # bending
  dΩs = Measure(Ω, 1)  # shear (reduced)

  # --- Function spaces (scalar components combined as a MultiField) ---
  # Unknowns: (u_x, u_y, w, θx, θy)
  reffe_scalar = ReferenceFE(lagrangian, Float64, 1)
  reffe_vector = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)

  g_zeroV(x) = VectorValue(0.0, 0.0)
  g_zero(x) = 0.0
  g_th(x) = VectorValue(0.0, pi/10) # prescribed rotation on "left" (as in your code)

  V_u = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["right"])
  U_u = TrialFESpace(V_u, g_zeroV)

  V_w = TestFESpace(model, reffe_scalar; conformity=:H1, dirichlet_tags=["right"])
  U_w = TrialFESpace(V_w, g_zero)

  V_th = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["left"])
  U_th = TrialFESpace(V_th, g_th)

  V = MultiFieldFESpace([V_u, V_w, V_th])
  U = MultiFieldFESpace([U_u, U_w, U_th])

  # --- Material (plane-stress) ---
  μ   = E/(2*(1+ν))
  λps = 2*μ*ν/(1-ν)           # effective λ for plane stress
  G   = μ
  Sshear = κs*G*h             # κ G h

  # Plane-stress 4th-order contraction:
  # (C:A):B = λ tr(A) tr(B) + 2μ A:B
  innerC(A,B) = λps*tr(A)*tr(B) + 2*μ*inner(A,B)

  # Helpers
  εm(u)   = ε(u)                      # small-strain sym grad (used inside GL)
  κb(θ)   = ε(θ)                      # curvatures (linear bending proxy)
  γs(w,θ) = ∇(w) - θ                  # shear strains γxz, γyz

  # --- Total Lagrangian Green–Lagrange membrane strain and its variation ---
  # E = sym(∇u) + 1/2 [ (∇u)^T ∇u + ∇w ⊗ ∇w ]
  function E_GL(u,w)
    gu = ∇(u)               # 2x2
    gw = ∇(w)               # 2
    E_lin  = εm(u)
    quad   = transpose(gu) ⋅ gu + (gw ⊗ gw)
    return E_lin + 0.5*(quad + transpose(quad)) # ensure symmetric
  end

  # δE = sym(∇v) + 1/2 [ (∇v)^T ∇u + (∇u)^T ∇v + ∇vw ⊗ ∇w + ∇w ⊗ ∇vw ]
  function dE_GL(u,w,v,vw)
    gu = ∇(u); gw = ∇(w)
    gv = ∇(v); gvw = ∇(vw)
    term_lin  = εm(v)
    term_quad = transpose(gv) ⋅ gu + transpose(gu) ⋅ gv + (gvw ⊗ gw) + (gw ⊗ gvw)
    return term_lin + 0.5*(term_quad + transpose(term_quad))
  end

  # --- Central Gaussian transverse load (approximate point force) ---
  P  = -100.0                 # total force (negative = downward)
  x0 = (@SVector [Lx/2, Ly/2])
  σ  = min(Lx,Ly)/40          # width of Gaussian; smaller ⇒ more concentrated
#  pfun(x) = P/(2π*σ^2) * exp(-0.5*((x[1]-x0[1])^2 + (x[2]-x0[2])^2)/(σ^2))
  pfun(x) = 0.0

  # --- Nonlinear residual and consistent tangent (Newton) ---
  # Residual R(u,w,θ; v,vw,ψ) = Rm + Rb + Rs - L
  function res((u,w,θ),(v,vw,ψ))
    Rm = ∫( h * innerC( E_GL(u,w), dE_GL(u,w,v,vw) ) )dΩm
    Rb = ∫( (h^3/12) * ( λps*tr(κb(θ))*tr(κb(ψ)) + 2*μ*inner(κb(θ), κb(ψ)) ) )dΩb
    Rs = ∫( Sshear * inner( γs(w,θ), γs(vw,ψ) ) )dΩs
    L  = ∫( pfun * vw )dΩm
    return Rm + Rb + Rs - L
  end

  # Jacobian: derivative of residual in direction (du,dw,dθ)
  function jac((u,w,θ),(du,dw,dθ),(v,vw,ψ))
    Jm = ∫( h * innerC( dE_GL(u,w,du,dw), dE_GL(u,w,v,vw) ) )dΩm
    Jb = ∫( (h^3/12) * ( λps*tr(κb(dθ))*tr(κb(ψ)) + 2*μ*inner(κb(dθ), κb(ψ)) ) )dΩb
    Js = ∫( Sshear * inner( ∇(dw) - dθ, ∇(vw) - ψ ) )dΩs
    return Jm + Jb + Js
  end

  op = FEOperator(res, jac, U, V)

  # --- Solve nonlinear problem ---
  uh, wh, θh = solve(op)

  # --- Output ---
  writevtk(Ω, "gridap_large_strain_shell";
    cellfields = [
      "u" => uh,
      "w" => wh,
      "θ" => θh
    ])
end

main()