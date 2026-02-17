using Gridap                           # Core FEM library for defining models, spaces, forms, and solvers
using LinearAlgebra                    # Standard Julia linear algebra utilities
using StaticArrays                     # Stack-allocated fixed-size vectors/matrices (used for efficiency)
using Gridap.TensorValues: symmetric_part  # Import symmetric tensor projection (keeps SymTensorValue)
#using Gridap.Algebra                   # Nonlinear solvers (NLSolver), FE solver wrappers, etc.
#using LineSearches: BackTracking      # (Optional) Line search strategy from LineSearches.jl if needed

function main()                        # Entry point for the script
  # --- Geometry & mesh ---
  nels = (50,50)                       # Number of elements in x and y directions (uniform Cartesian mesh)
  Lx, Ly = 10.0, 10.0                  # Plate dimensions along x and y (mid-surface size)
  h  = 1.0                             # Plate thickness
  E  = 210.0                           # Young's modulus (consistent units)
  ν  = 0.3                             # Poisson’s ratio
  κs = 5/6                             # Shear correction factor for Reissner–Mindlin plates

  model = CartesianDiscreteModel((0.0,Lx, 0.0,Ly), nels)  # Build structured mesh over [0,Lx]×[0,Ly]
  labels = get_face_labeling(model)     # Access face labeling object to define boundary tags
  # In Gridap's Cartesian models
  add_tag_from_tags!(labels,"left", [1, 3, 7])    # Tag "left" boundary faces (IDs depend on Gridap’s scheme)
  add_tag_from_tags!(labels,"right", [2, 4, 8])   # Tag "right" boundary faces
  add_tag_from_tags!(labels,"bottom",[1, 2, 5])   # Tag "bottom" boundary faces
  add_tag_from_tags!(labels,"top", [3, 4, 6])     # Tag "top" boundary faces

  Ω  = Triangulation(model)             # Build the cell triangulation over the domain
  # Under-integration for shear to reduce locking
  dΩm = Measure(Ω, 2)                   # Integration measure for membrane terms (full integration)
  dΩb = Measure(Ω, 2)                   # Integration measure for bending terms (full integration)
  dΩs = Measure(Ω, 1)                   # Reduced integration for shear (to alleviate shear locking)

  # --- Function spaces (scalar components combined as a MultiField) ---
  # Unknowns: (u_x, u_y, w, θx, θy)
  reffe_scalar = ReferenceFE(lagrangian, Float64, 1)                    # P¹ scalar FE for w
  reffe_vector = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)    # P¹ vector FE (2D) for u, θ

  g_zeroV(x) = VectorValue(0.0, 0.0)   # Zero vector Dirichlet data (for u on the right side)
  g_zero(x)  = 0.0                     # Zero scalar Dirichlet data (for w on the right side)
  g_th(x)    = VectorValue(0.0, pi/20) # Prescribed rotation θ on "left" boundary (θx=0, θy=π/20)

  V_u = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["right"])  # H¹ test space for u, clamp on right
  U_u = TrialFESpace(V_u, g_zeroV)     # Trial space for u with Dirichlet data g_zeroV

  V_w = TestFESpace(model, reffe_scalar; conformity=:H1, dirichlet_tags=["right"])  # H¹ test space for w, clamp on right
  U_w = TrialFESpace(V_w, g_zero)      # Trial space for w with Dirichlet data g_zero

  V_th = TestFESpace(model, reffe_vector; conformity=:H1, dirichlet_tags=["left"])  # H¹ test space for θ, prescribe on left
  U_th = TrialFESpace(V_th, g_th)      # Trial space for θ with Dirichlet data g_th

  V = MultiFieldFESpace([V_u, V_w, V_th])  # Block test space for (u, w, θ)
  U = MultiFieldFESpace([U_u, U_w, U_th])  # Block trial space for (u, w, θ)

  # --- Material (plane-stress) ---
  μ   = E/(2*(1+ν))                    # Shear modulus G = E/(2(1+ν))
  λps = 2*μ*ν/(1-ν)                    # Plane-stress effective Lamé λ (for 2D plate membrane)
  G   = μ                              # Shear modulus alias
  Sshear = κs*G*h                      # Shear stiffness factor κ·G·h (Reissner–Mindlin shear)

  # Plane-stress 4th-order contraction:
  # (C:A):B = λ tr(A) tr(B) + 2μ A:B
  @inline innerC(A,B) = λps*tr(A)*tr(B) + 2*μ*inner(A,B)  # Isotropic plane-stress bilinear form

  # Helpers
  @inline εm(u)   = ε(u)               # Small-strain symmetric gradient of in-plane displacement u
  @inline κb(θ)   = ε(θ)               # Linear curvature proxy via symmetric gradient of θ
  @inline γs(w,θ) = ∇(w) - θ           # Transverse shear strain vector γ = ∇w − θ

  # --- Total Lagrangian Green–Lagrange membrane strain and its variation ---
  # Keep outputs SymTensorValue to exploit symmetric algebra
  @inline function E_GL(u,w)           # Green–Lagrange membrane strain for (u,w)
    gu = ∇(u)                          # Gradient of in-plane displacement u (2×2)
    gw = ∇(w)                          # Gradient of transverse displacement w (2×1)
    E_lin  = εm(u)                     # Linear strain (symmetric)
    quad   = transpose(gu) ⋅ gu + (gw ⊗ gw)  # Quadratic terms from large strains (non-symmetric TensorValue)
    return E_lin + symmetric_part(quad)      # Symmetric part → SymTensorValue (performance-friendly)
  end

  @inline function dE_GL(u,w,v,vw)     # Variation of GL strain δE in direction (v,vw)
    gu = ∇(u); gw = ∇(w)               # Current gradients (state)
    gv = ∇(v); gvw = ∇(vw)             # Variation gradients (test directions)
    term_lin  = εm(v)                  # Linear variation (symmetric)
    term_quad = transpose(gv) ⋅ gu + transpose(gu) ⋅ gv + (gvw ⊗ gw) + (gw ⊗ gvw)  # Geometric nonlinearity
    return term_lin + symmetric_part(term_quad)  # Symmetric projection to keep SymTensorValue
  end

  # --- Load (skip if zero to avoid useless loops) ---
  P  = -1.0                         # Total downward load (if using Gaussian)
  x0 = (@SVector [Lx/2, Ly/2])        # Load center at plate center
  σ  = min(Lx,Ly)/40                  # Gaussian width (controls concentration)
  pfun(x) = P/(2π*σ^2) * exp(-0.5*((x[1]-x0[1])^2 + (x[2]-x0[2])^2)/(σ^2))  # Distributed load

  # --- Nonlinear residual and consistent tangent (Newton) ---
  function res((u,w,θ),(v,vw,ψ))       # Residual form R(U; V) = membrane + bending + shear − load
    Rm = ∫( h * innerC( E_GL(u,w), dE_GL(u,w,v,vw) ) )dΩm     # Membrane contribution (GL kinematics)
    Rb = ∫( (h^3/12) * ( λps*tr(κb(θ))*tr(κb(ψ)) + 2*μ*inner(κb(θ), κb(ψ)) ) )dΩb  # Bending (linear)
    Rs = ∫( Sshear * inner( γs(w,θ), γs(vw,ψ) ) )dΩs          # Shear with reduced integration
    L  = ∫(pfun * vw)dΩm
    return Rm + Rb + Rs + L                                      # No external load term subtracted here
  end

  function jac((u,w,θ),(du,dw,dθ),(v,vw,ψ))  # Consistent tangent (Jacobian) dR(U)[dU]·V
    Jm = ∫( h * innerC( dE_GL(u,w,du,dw), dE_GL(u,w,v,vw) ) )dΩm  # Membrane tangent
    Jb = ∫( (h^3/12) * ( λps*tr(κb(dθ))*tr(κb(ψ)) + 2*μ*inner(κb(dθ), κb(ψ)) ) )dΩb  # Bending tangent
    Js = ∫( Sshear * inner( ∇(dw) - dθ, ∇(vw) - ψ ) )dΩs          # Shear tangent
    return Jm + Jb + Js                                           # Total Jacobian
  end

  op = FEOperator(res, jac, U, V)       # Assemble nonlinear FE operator from residual/Jacobian and spaces

  # --- Solve nonlinear problem ---
  nls = NLSolver(                       # Configure the nonlinear solver (NLsolve backend)
    show_trace=true,                    # Print nonlinear iteration info
    method=:newton,                     # Use Newton's method
    xtol = 1e-6,                        # Tolerance on update (||Δx||)
    ftol = 1e-3                         # Tolerance on residual (||F||)
    #linesearch=BackTracking()          # Optional line search; uncomment if using LineSearches.jl
  )

  solver = FESolver(nls)                # Wrap nonlinear solver for FE problems
  uh, wh, θh = solve(solver, op)        # Solve the nonlinear system → solution fields (u, w, θ)

  # --- Output ---
  writevtk(Ω, "gridap_large_strain_shell";   # Export results to VTK for Paraview/VisIt
    cellfields = [
      "u" => uh,                        # In-plane displacement vector field
      "w" => wh,                        # Transverse displacement scalar field
      "θ" => θh                         # Rotation vector field
    ])
end

main()                                  # Run the main function