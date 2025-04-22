
# Comprehensive Elaboration of the Cahn–Hilliard Simulation Code

## 1. Overview and Purpose

The presented code implements a two-dimensional simulation of the Cahn–Hilliard equation, a model commonly used to describe phase separation and coarsening in binary mixtures. In the simulation, two fields are considered:
- **Phase field** $(c)$: Represents the concentration or order parameter of one of the components.
- **Chemical potential** $(\mu)$: Acts as the driving force for phase separation, derived from a free energy functional.

The simulation is performed on a unit square $\([0,1] \times [0,1]\)$ using a finite element method (FEM) framework provided by Gridap in Julia. Time integration is carried out via the Theta method (a generalization of Crank–Nicolson with $\( \theta=0.5 \)$ ), and the nonlinear system is solved using Newton’s method with automatic differentiation (AD) to compute the required Jacobian matrices.

---

## 2. Mesh and Finite Element Space Setup

### 2.1. Mesh Definition

- **Domain and Partitioning:**  
  The simulation domain is defined as the unit square:
  ```julia
  domain = (0.0, 1.0, 0.0, 1.0)
  ```
  This domain is discretized into a structured Cartesian grid of $\(96 \times 96\)$ cells:
  ```julia
  partition = (96, 96)
  model = CartesianDiscreteModel(domain, partition)
  ```

### 2.2. Finite Element Spaces

- **Element Order:**  
  Linear elements (first order) are used for simplicity and computational efficiency:
  ```julia
  order = 1
  ```

- **Phase Field $\( c \)$ and its Transient Trial Space:**  
  A Lagrangian finite element is defined and then used to set up the $\( H^1 \)$-conforming finite element space for the phase field:
  ```julia
  reffe_c = ReferenceFE(lagrangian, Float64, order)
  V = FESpace(model, reffe_c, conformity=:H1)
  U = TransientTrialFESpace(V)
  ```

- **Chemical Potential $\( \mu \)$ and its Trial Space:**  
  Similarly, a finite element space is defined for the chemical potential:
  ```julia
  reffe_μ = ReferenceFE(lagrangian, Float64, order)
  Q = FESpace(model, reffe_μ, conformity=:H1)
  P = TrialFESpace(Q)
  ```

- **Multi-Field Spaces for Coupled Variables:**  
  The two fields are bundled into multi-field spaces for both the trial and test functions:
  ```julia
  X = TransientMultiFieldFESpace([U, P])
  Y = MultiFieldFESpace([V, Q])
  ```

---

## 3. Quadrature and Numerical Integration

- **Integration Rule:**  
  For numerical integration over the mesh, a quadrature rule of degree 2 is applied:
  ```julia
  degree = 2
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  ```

---

## 4. Model Parameters and Energy Functional

- **Physical Parameters:**  
  The mobility $\( M \)$ and the interfacial (gradient penalty) parameter $\( \lambda \)$ are given by:
  ```julia
  M = 1.0
  λ = 1.0e-2
  ```

- **Free Energy Function $\( f(c) \)$ :**  
  A double-well potential is used to describe the free energy density, reflecting the two preferred states (phases) of the binary mixture:
  ```julia
  f(c) = 100*c^2*(1-c)^2
  ```

- **Derivative of Free Energy $\( f'(c) \)$ :**  
  The derivative with respect to $\( c \)$ , essential for deriving the chemical potential $\( \mu \)$ , is defined as:
  ```julia
  dfdc = (c) -> 100*(4*c^3 - 6*c^2 + 2*c)
  ```

---

## 5. Weak Formulation: Residual Definitions

The Cahn–Hilliard equations are converted into a weak (integral) form suitable for FEM. The weak formulation involves integrating against test functions, which leads to the definition of various residual terms:

- **Mass Term (Time Derivative):**  
  The term representing the time derivative of the phase field $\( c \)$ is written as:
  ```julia
  mass(t, (c, μ), (ct, μt), (dc, dμ)) = ∫(ct ⋅ dc) * dΩ
  ```
  An alternative semilinear version is also provided:
  ```julia
  mass_sl(t, (ct, μt), (dc, dμ)) = ∫(ct ⋅ dc) * dΩ
  ```

- **Diffusion Term for $\( \mu \)$ :**  
  The weak form includes a diffusive term involving the gradients of $\( \mu \)$ :
  ```julia
  res_ql1(t, μ, dc) = ∫(M*∇(μ)⋅∇(dc)) * dΩ
  ```

- **Coupling Term between $\( c \)$ and $\( \mu \)$ :**  
  This residual captures the relationship between the chemical potential and the phase field:
  ```julia
  res_ql2(t, (c, μ), dμ) = ∫(μ ⋅ dμ - dμ ⋅ dfdc(c) - λ*∇(dμ)⋅∇(c)) * dΩ
  ```

- **Combined Residual for the Quasilinear Problem:**  
  The full quasilinear residual is the sum of the diffusion and coupling components:
  ```julia
  res_ql(t, (c, μ), (dc, dμ)) = res_ql1(t, μ, dc) + res_ql2(t, (c, μ), dμ)
  ```

- **Alternative Nonlinear Residual Splitting:**  
  Another formulation splits the residual into:
  - $\( res1 \)$ : Incorporating the time derivative and diffusion, and
  - $\( res2 \)$ : Incorporating the chemical potential and its gradient energy.
  ```julia
  res1(t, (c, μ), (dc, dμ)) = ∫(∂t(c) ⋅ dc + M*∇(μ)⋅∇(dc)) * dΩ
  res2(t, (c, μ), (dc, dμ)) = ∫(μ ⋅ dμ - dfdc(c) ⋅ dμ - λ*∇(dμ)⋅∇(c)) * dΩ
  res(t, (c, μ), (dc, dμ)) = res1(t, (c, μ), (dc, dμ)) + res2(t, (c, μ), (dc, dμ))
  ```

---

## 6. Operator and Solver Configuration

### 6.1. Finite Element Operators

The weak formulations are encapsulated into finite element operators that use automatic differentiation (AD) for the efficient evaluation of the Jacobian:

- **Nonlinear Operator:**  
  Designed for solving the fully nonlinear system:
  ```julia
  tfeop_nl = TransientFEOperator(res, X, Y)
  ```

- **Quasilinear and Semilinear Operators:**  
  Alternative formulations that emphasize different aspects of the system’s mass term:
  ```julia
  tfeop_ql = TransientQuasilinearFEOperator(mass, res_ql, X, Y)
  tfeop_sl = TransientSemilinearFEOperator(mass_sl, res_ql, X, Y)
  ```

### 6.2. Solver Setup

- **Linear and Nonlinear Solvers:**  
  A linear solver based on LU decomposition is chosen:
  ```julia
  lin_solver = LUSolver()
  ```
  A nonlinear solver that employs Newton's method is configured:
  ```julia
  nl_solver = NLSolver(lin_solver, method=:newton, iterations=50, show_trace=true, ftol=1e-6)
  ```

- **Time-Stepping Scheme (Theta Method):**  
  The Theta method is used for temporal integration. With $\( \theta = 0.5 \)$ , this becomes the Crank–Nicolson method:
  ```julia
  Δt = 5 / 1000000      # Time step size.
  θ = 0.5
  solver = ThetaMethod(nl_solver, Δt, θ)
  t0, tF = 0.0, 50*Δt   # Simulation start and final times.
  ```

---

## 7. Initial Condition Setup

### 7.1. Defining the Initial Phase Field $\( c_0 \)$ 

A small random perturbation about a mean value of $0.63$ is introduced to mimic slight inhomogeneities:
```julia
function c0(x)
    0.63 + 0.02 * (0.5 - rand())
end
```

### 7.2. Defining the Initial Chemical Potential $\( \mu_0 \)$ 

Using the free energy density and its derivative, the initial chemical potential is computed as:
```julia
function μ0(x)
    f(c0(x)) - λ * laplacian(c0)(x)
end
```

### 7.3. Interpolation to FE Space

The initial multi-field state, combining $\( c_0 \)$ and $\( \mu_0 \)$ , is interpolated onto the finite element space:
```julia
X0 = X(t0)
xh0 = interpolate_everywhere([c0, μ0], X0)
```

---

## 8. Time Integration and Solving

The Theta method advances the solution in time. At each time step, the transient solver uses the nonlinear operator and the current state to compute the next state. The collection of solutions for each time step is stored in `Xh`:
```julia
Xh = solve(solver, tfeop_nl, t0, tF, xh0)
```
Key points:
- Newton’s method is employed iteratively at each time step.
- Automatic differentiation ensures efficient computation of the Jacobian.
- The time-stepping scheme provides a second-order accurate integration.

---

## 9. Output Management and Visualization

### 9.1. File Clean-Up

Before saving new VTK files (used for visualization in tools like ParaView), the code ensures that any old files are removed:
```julia
function delete_vtk_files(folder_path::String, extension::String)
    if !isdir(folder_path)
        error("Folder '$folder_path' does not exist")
    end
    files = readdir(folder_path)
    for file in files
        if endswith(file, extension)
            full_path = joinpath(folder_path, file)
            rm(full_path)
            println("Deleted: $full_path")
        end
    end
end
```

### 9.2. Output Directory Setup and Writing VTK Files

The temporary output directory is verified (and created if it does not exist). Then, a PVD file is generated which aggregates the time-dependent VTK outputs:
```julia
if !isdir("tmp/tmp_CahnHilliard")
    mkdir("tmp/tmp_CahnHilliard")
else
    delete_vtk_files("tmp/tmp_CahnHilliard", ".vtu")
end

@time createpvd("tmp/results_CahnHilliard") do pvd
    pvd[0] = createvtk(Ω, "tmp/tmp_CahnHilliard/results_0.vtu", cellfields=["c" => xh0[1], "μ" => xh0[2]])
    for (tn, (cn, μn)) in Xh
        @show tn
        pvd[tn] = createvtk(Ω, "tmp/tmp_CahnHilliard/results_$tn.vtu", cellfields=["c" => cn, "μ" => μn])
    end
end
```

---

## 10. Physics and Mathematical Formulation

### 10.1. The Cahn–Hilliard Equation

The Cahn–Hilliard equation models the process of phase separation in a binary mixture. It describes how a conserved order parameter $\( c \)$ (for instance, the concentration of one component) evolves over time due to the minimization of a free energy functional $\( F(c) \)$. The general form of the Cahn–Hilliard equation is:
 $$\frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \mu \right)$$
where:
- $\( M \)$ is the mobility, which governs the rate of mass transport.
- $\( \mu \)$ is the chemical potential, defined as the variational derivative of the free energy with respect to $\( c \)$ :
 $$\mu = \frac{\delta F}{\delta c}$$.

### 10.2. Free Energy Functional

A common choice for the free energy is a combination of a local free energy density $\( f(c) \)$ and a gradient energy term. This is expressed as:
$$F(c) = \int \left( f(c) + \frac{\lambda}{2} \left\lvert \nabla c \right\rvert^2 \right) dx$$ , 
where:
- $\( f(c) \)$ is typically chosen as a double-well potential with two minima, representing two preferred phases.
- $\( \lambda \)$ is a parameter controlling the penalty on the gradients of $\( c \)$, influencing the interfacial energy and the smoothness of the interface between phases.

For this simulation:
- The double-well potential is defined as:
$f(c) = 100c^2(1-c)^2$ ,
  having minima at $\( c \approx 0 \)$ and $\( c \approx 1 \)$.
- The derivative with respect to $\( c \)$ is:
  $f'(c) = 100(4c^3 - 6c^2 + 2c)$.
- The chemical potential $\( \mu \)$ is then given by:
  $\mu = f'(c) - \lambda \Delta c$, where $\( \Delta c \)$ is the Laplacian of $\( c \)$.

### 10.3. Weak Formulation and Numerical Approximation

The finite element formulation involves rewriting the Cahn–Hilliard equation in its weak form. This requires multiplying by test functions and integrating over the domain. The weak formulation naturally leads to integrals that contain:
- The time derivative $\( \partial_t c \)$,
- Diffusion-like terms involving the gradients of $\( \mu \)$ and $\( c \)$,
- Coupling terms that stem from the variation of the free energy functional.

In the code, the residuals are crafted to represent these weak formulations, accounting for both the mass conservation and the energetics of the system.

### 10.4. Solver and Time Integration

The nonlinear solver (Newton’s method) along with automatic differentiation ensures that the Jacobian of the discretized system is computed accurately, allowing for robust convergence. The Theta method (with $\( \theta=0.5 \))$ strikes a balance between stability and accuracy, making it suitable for a system with stiff dynamics such as the Cahn–Hilliard equation.

---

## 11. Results

The initial configuration $c$:

<img width="493" alt="init_c" src="https://github.com/user-attachments/assets/b736ec6f-673b-44d9-bd69-ada8153757d6" />

The final configuration $c$:

<img width="498" alt="final_c" src="https://github.com/user-attachments/assets/7376315c-6d8b-4c2d-91ce-a801d2b2f025" />

## 12. Concluding Remarks

This code provides a complete pipeline for simulating phase separation using the Cahn–Hilliard model:
- **Mesh and FE Spaces:** A Cartesian grid and appropriate finite element spaces are defined.
- **Physics and Mathematics:** The underlying physics is encapsulated via the free energy functional and the Cahn–Hilliard equation.
- **Numerical Formulation:** A weak formulation is employed, leading to residuals that capture the dynamics and energetics.
- **Solver Configuration:** A robust nonlinear solver combined with a time-stepping scheme ensures accurate evolution over time.
- **Output:** Results are exported for visualization, enabling further analysis of the phase separation process.

This documentation should serve as a clear reference for understanding both the physics and implementation details of the simulation, as well as for further extension or modification of the code for related phase-field models.
