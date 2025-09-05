# Linear shell (Reissner–Mindlin) in Gridap.jl matching the Ferrite example setup
# - Flat rectangular plate mid-surface (10 x 10)
# - Thickness h = 1.0
# - E = 210, ν = 0.3, shear correction κ = 5/6
# - BCs:
#     left:  u_x = 0, w = 0, θx = 0, θy = 0
#     right: u_x = 0, w = 0, θx = 0, θy = π/10
#   (and u_y fixed on left to avoid in-plane rigid motion; this mirrors the corner pin in the Ferrite demo)

using Gridap
using LinearAlgebra
using StaticArrays
# Explicitly import FESpaces helpers for wider Gridap version compatibility
# using Gridap.FESpaces: TestFESpace, TrialFESpace, TrialFunction, TestFunction, MultiFieldFESpace, trial_and_test

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
#   g_th(x) = VectorValue(0.0, (abs(x[1]-Lx) < 1.0e-12) ? (pi/10) : 0.0)
    g_th(x) = VectorValue(0.0, pi/10)

    V_u = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["right"])
    U_u = TrialFESpace(V_u, g_zeroV)

    V_w = TestFESpace(model, reffe_scalar; conformity=:H1, dirichlet_tags=["right"])
    U_w = TrialFESpace(V_w, g_zero)

    V_th = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["left"])
    U_th = TrialFESpace(V_th, g_th)

    V = MultiFieldFESpace([V_u, V_w, V_th])
    U = MultiFieldFESpace([U_u, U_w, U_th])

  # --- Material operators (Voigt 2D: [εxx, εyy, γxy]) ---
  A = (E/(1-ν^2))*[1.0  ν    0.0;
                   ν    1.0  0.0;
                   0.0  0.0  (1-ν)/2]
  Cm = h*A                    # membrane (per area)
  Cb = (h^3/12)*A             # bending (per area)
  G  = E/(2*(1+ν))
  Sshear = κs*G*h             # shear stiffness (scalar) times I₂

  # --- Material (plane-stress) ---
  μ   = E/(2*(1+ν))
  λps = 2*μ*ν/(1-ν)          # effective λ for plane stress
  Sshear = κs*μ*h            # κ G h, with G = μ

  # Helpers
  voigt2(e) = @SVector [e[1], e[2], 2*e[3]] # SymTensor → engineering strain
  εm(u)   = ε(u)                                   # membrane strains
  κb(θ)   = ε(θ)                                   # curvatures
  γs(w,θ) = ∇(w) - θ                               # shear strains γxz,γyz

  # --- Central Gaussian transverse load (approximate point force) ---
  P  = -100.0                 # total force (negative = downward)
  x0 = (@SVector [Lx/2, Ly/2])
  σ  = min(Lx,Ly)/40          # width of Gaussian; smaller ⇒ more concentrated
  pfun(x) = P/(2π*σ^2) * exp(-0.5*((x[1]-x0[1])^2 + (x[2]-x0[2])^2)/(σ^2))

  # Bilinear form
#   a_m((u, w, θ), (v, vw, ψ)) = ∫( (voigt2(εm(u)) ⋅ (Cm*voigt2(εm(v)))) )dΩm
#   a_b((u, w, θ), (v, vw, ψ)) = ∫( (voigt2(κb(θ))   ⋅ (Cb*voigt2(κb(ψ))))   )dΩb
#   a_s((u, w, θ), (v, vw, ψ)) = ∫( (γs(w,θ) ⋅ (Sshear*γs(vw,ψ))) )dΩs

    a_m((u, w, θ), (v, vw, ψ)) = ∫( h * ( λps*tr(ε(u))*tr(ε(v)) + 2*μ*inner(ε(u), ε(v)) ) )dΩm
    a_b((u, w, θ), (v, vw, ψ)) = ∫( (h^3/12) * ( λps*tr(ε(θ))*tr(ε(ψ)) + 2*μ*inner(ε(θ), ε(ψ)) ) )dΩb
    a_s((u, w, θ), (v, vw, ψ)) = ∫( Sshear * inner( γs(w,θ), γs(vw,ψ) ) )dΩs

  a((u, w, θ), (v, vw, ψ)) = a_m((u, w, θ), (v, vw, ψ)) + a_b((u, w, θ), (v, vw, ψ)) +
   a_s((u, w, θ), (v, vw, ψ))
  l((v, vw, ψ)) = ∫( pfun * vw )dΩm

#   op  = AffineFEOperator(a,l,U,V)
    op = FEOperator(a, U, V)
  uh, wh, θh = solve(op)


  # --- Output ---
  writevtk(Ω, "gridap_linear_shell";
    cellfields = [
      "u"=>uh,
      "w"=>wh,
      "θ"=>θh
    ])
end

main()
