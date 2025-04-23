# Brusselator PDE Solver Using Gridap

This document explains the code that sets up and solves a time-dependent reaction-diffusion system—a Brusselator model—using the Gridap package in Julia. The code uses finite element methods (FEM) for spatial discretization and a Theta (Crank-Nicolson) method for time integration.

## 1. Overview

The goal is to solve a coupled system of partial differential equations (PDEs) for the unknown functions $u(x,t)$ and $v(x,t)$ in a periodic two-dimensional domain. The system models a reaction-diffusion process and includes non-linear reaction terms.

## 2. Domain and Mesh

- **Domain:** A rectangular region $[0,1] \times [0,1]$  
- **Mesh:** The domain is partitioned into a $32 \times 32$ Cartesian grid.  
- **Periodicity:** The domain is periodic in both $x$ and $y$ directions.

```julia
dimain = (0, 1, 0, 1)
partition = (32, 32)
model = CartesianDiscreteModel(dimain, partition, isperiodic=(true, true))
```

## 3. Finite Element Spaces

The code uses first-order Lagrangian (linear) finite elements to approximate the solution in the Sobolev space $H^1$. Two fields $u$ and $v$ are defined:

- **Reference Finite Element:** Defined using Lagrangian elements of order 1.
- **Trial and Test Spaces:**  
  - $U$ and $V$ are built as transient trial spaces (for time-dependent problems).  
  - Corresponding test spaces $U\\\_$ and $V\\\_$ are used to form the weak formulation.

```julia
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
U_ = FESpace(model, reffe, conformity=:H1)
U = TransientTrialFESpace(U_)
V_ = FESpace(model, reffe, conformity=:H1)
V = TransientTrialFESpace(V_)
X = TransientMultiFieldFESpace([U, V])
Y = MultiFieldFESpace([U_, V_])
```

## 4. Weak Formulation and Residual Equations

### Governing Equations

The model consists of two coupled PDEs. In strong form, they are written as:

#### For $u$:
$\frac{\partial u}{\partial t} - g_1(u,v) - \alpha \Delta u = f(t,x)$

with the reaction function:
$g_1(u,v) = 1 + u^2v - 4.4u$,
and a forcing function $f(t,x)$ defined by:

![image](https://github.com/user-attachments/assets/4fa778bd-685e-4ca4-8912-e174febd544e)

#### For $v$:
$\frac{\partial v}{\partial t} - g_2(u,v) - \alpha \Delta v = 0$,
with the reaction function:
$g_2(u,v) = 3.4u - u^2v$.

### Weak Formulation

Multiplying the equations by test functions (denoted $u^\*$ and $v^\*$) and integrating over the domain $\Omega$ leads to:

#### $u$-Equation:
$\int_{\Omega} \left( \frac{\partial u}{\partial t}. u^* - g_1(u,v). u^* + \alpha \nabla u \cdot \nabla u^* - f(t,x). u^* \right) d\Omega = 0$,

#### $v$-Equation:
$\int_{\Omega} \left( \frac{\partial v}{\partial t}. v^* - g_2(u,v). v^* + \alpha \nabla v \cdot \nabla v^* \right) d\Omega = 0$.

In the code, these equations are defined through the residual functions `res1` and `res2` and then combined:

```julia
res1(t, (u, v), (u_, v_)) = ∫(∂t(u)*u_ - g1(u, v)*u_ + α*∇(u)⋅∇(u_) - f_t(t)*u_) * dΩ
res2(t, (u, v), (u_, v_)) = ∫(∂t(v)*v_ - g2(u, v)*v_ + α*∇(v)⋅∇(v_)) * dΩ
res(t, (u, v), (u_, v_)) = res1(t, (u, v), (u_, v_)) + res2(t, (u, v), (u_, v_))
```

## 5. Initial Conditions

The initial states for $u$ and $v$ are provided as functions that ensure the solution vanishes at the boundaries:

```julia
u0(x) = 22.0 * (x[2]*(1-x[2]))^(3/2)
v0(x) = 27.0 * (x[1]*(1-x[1]))^(3/2)
```

These are interpolated over the finite element space to set up the initial condition.

## 6. Time Integration and Nonlinear Solver

### Time-Stepping

- **Method:** Theta method (with $\theta = 0.5$) corresponding to the Crank-Nicolson scheme.
- **Time Step:** $\Delta t = 0.5$
- **Time Interval:** $t \in [0, 11.5]$

```julia
Δt = 0.5      # Time step size.
θ = 0.5       # Crank-Nicolson scheme if θ = 0.5.
```

### Solvers

- **Nonlinear Solver:** Newton method is used to solve the nonlinear residual, with a maximum of 50 iterations and convergence tolerance $10^{-6}$.
- **Linear Solver:** LU factorization handles the linear sub-problems.

```julia
lin_solver = LUSolver()
nl_solver = NLSolver(lin_solver, method=:newton, iterations=50, show_trace=true, ftol=1e-6)
solver = ThetaMethod(nl_solver, Δt, θ)
t0, tF = 0.0, 11.5   # Simulation start and final times.
```

The initial finite element state is obtained and used to solve the PDE:

```julia
X0 = X(t0)
xh0 = interpolate_everywhere([u0, v0], X0)
Xh = solve(solver, tfeop_nl, t0, tF, xh0)
```

## 7. Result Output and Visualization

The solution at each time step is saved in VTK format, which can later be visualized using a VTK viewer (e.g., ParaView). The code checks and creates the output directory if necessary, and writes out the simulation results.

```julia
if !isdir("tmp/tmp_Brusselator")
    mkdir("tmp/tmp_Brusselator")
end

@time createpvd("tmp/tmp_Brusselator") do pvd
    pvd[0] = createvtk(Ω, "tmp/tmp_Brusselator/results_0.vtu", cellfields=["u" => xh0[1], "v" => xh0[2]])
    for (tn, (un, vn)) in Xh
        @show tn
        pvd[tn] = createvtk(Ω, "tmp/tmp_Brusselator/results_$tn.vtu", cellfields=["u" => un, "v" => vn])
    end
end
```

## 8. Summary of Equations and Workflow

- **Model PDEs:**
  - For $u$:
    $\frac{\partial u}{\partial t} - \left(1 + u^2v - 4.4u\right) - \alpha.\Delta u = f(t,x)$
  - For $v$:
    $\frac{\partial v}{\partial t} - \left(3.4u - u^2v\right) - \alpha.\Delta v = 0$.

- **Weak Formulation:**
  - $u$-Equation:
    $\int_\Omega \left( \frac{\partial u}{\partial t}. u^* - (1 + u^2v - 4.4u). u^* + \alpha. \nabla u \cdot \nabla u^* - f(t,x). u^* \right) d\Omega = 0$,
  - $v$-Equation:
    $\int_\Omega \left( \frac{\partial v}{\partial t}. v^* - (3.4u - u^2v). v^* + \alpha. \nabla v \cdot \nabla v^* \right) d\Omega = 0$.

- **Time Integration:**
  - **Method:** Crank-Nicolson ($\theta = 0.5$)
  - **Time Step:** $\Delta t = 0.5$
  - **Time Range:** $t = 0$ to $t = 11.5$

- **Numerical Solution:**  
  The nonlinear problem is solved using a Newton method (linearized with LU decomposition) at each time step, starting from the interpolated initial conditions.

- **Output:**  
  VTK files are generated for each time step, allowing for post-processing and visualization of the solution fields $u$ and $v$.
