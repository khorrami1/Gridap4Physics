
using Gridap
using Gridap.TensorValues
using GridapGmsh
using GridapGmsh.gmsh
using Plots

# input parameters
const T0 = 0
const TAppMax = T0+1.0e-6
const delt = 0.1
const tMax = 10.1
const uMax = 0.035e-3
AppVel = uMax/tMax
uMin = 0
uTran = uMax

function Tfun(u)  
    if u <= uTran
      return ((TAppMax - T0)/uTran)*u + T0
    else
     return  TAppMax
    end
end
plot(Tfun,0,uMax)

uAppVec = range(0,uMax,length = Int64(floor(tMax/delt)))

AppTOption = 2 ## 1 for smooth and otherwise linear than constant

if AppTOption == 1
    TAppVec = smoothT.(uAppVec)
  else
    TAppVec = Tfun.(uAppVec) 
end 

I2 = SymTensorValue{2,Float64}(1.0,0.0,1.0)
I4 = I2⊗I2
I4_sym = one(SymFourthOrderTensorValue{2,Float64})
I4_vol = (1.0/2)*I4
I4_dev = I4_sym - I4_vol

# Creating the geometry
const L = 0.05
const lsp = L/100
const eps = L/100
const Lc = 0.2*L 
const beta = 3*π/4
const Lcx = -Lc*cos(beta)
const Lcy = Lc*sin(beta)
const hfc = lsp/4
const hf = lsp/4
const h = L/8
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h, 1)  
gmsh.model.geo.addPoint(L, 0.0, 0.0, h, 2) 
gmsh.model.geo.addPoint(L, L-eps, 0.0,h, 3) 
gmsh.model.geo.addPoint(L-Lcx, L-eps+Lcy, 0.0, hfc, 4)
gmsh.model.geo.addPoint(L-Lcx+eps, L+Lcy,0.0, hfc, 5)
gmsh.model.geo.addPoint(L+eps, L, 0.0,h, 6)
gmsh.model.geo.addPoint(2*L, L,0.0, h, 7)
gmsh.model.geo.addPoint(2*L, 2*L, 0.0, h, 8)
gmsh.model.geo.addPoint(L, 2*L, 0.0, h, 9)
gmsh.model.geo.addPoint(L, 3*L, 0.0, h, 10)
gmsh.model.geo.addPoint(0, 3*L, 0.0, h, 11)
gmsh.model.geo.addPoint(0, 2*L, 0.0, hf, 12)
gmsh.model.geo.addPoint(-L, 2*L, 0.0, h, 13)
gmsh.model.geo.addPoint(-L, L, 0.0, h, 14)
gmsh.model.geo.addPoint(0, L, 0.0, h, 15)
gmsh.model.geo.addPoint(0, 1.15*L, 0.0, h, 16)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 9, 8)
gmsh.model.geo.addLine(9, 10, 9)
gmsh.model.geo.addLine(10, 11, 10)
gmsh.model.geo.addLine(11, 12, 11)
gmsh.model.geo.addLine(12, 13, 12)
gmsh.model.geo.addLine(13, 14, 13)
gmsh.model.geo.addLine(14, 15, 14)
gmsh.model.geo.addLine(15, 1, 15)
gmsh.model.geo.addLine(4, 16, 151)

gmsh.model.geo.addCurveLoop([12,13,14,15,1,2,3,4,5,6,7,8,9,10,11],1) 
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.addPhysicalGroup(2, [1],1)
gmsh.model.addPhysicalGroup(1, [1],1)
gmsh.model.addPhysicalGroup(1, [10],2)
gmsh.model.addPhysicalGroup(1, [13],3)
gmsh.model.addPhysicalGroup(1, [7],4)
gmsh.model.setPhysicalName(2, 1, "Domain")
gmsh.model.setPhysicalName(1, 1, "BottomEdge")
gmsh.model.setPhysicalName(1, 2, "TopEdge")
gmsh.model.setPhysicalName(1, 3, "LeftEdge")
gmsh.model.setPhysicalName(1, 4, "RightEdge")

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", [151])

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", hf)
gmsh.model.mesh.field.setNumber(2, "LcMax", h)
gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0*Lcy)
gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0*Lc)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("CrushiformShapeWithInclinedCrack(Mechanical).msh")
gmsh.finalize()

model = GmshDiscreteModel("CrushiformShapeWithInclinedCrack(Mechanical).msh")
writevtk(model,"CrushiformShapeWithInclinedCrack(Mechanical)")

using Gridap.Geometry
labels = get_face_labeling(model)
dimension = 2
mat_tags = get_face_tag(labels,dimension)

const Mat_tag = get_tag_from_name(labels,"Domain")

const E_mat = 218.4e3
const ν_mat = 0.2

# Input fracture parameters
const Gc = 2.0e-4
const η = 1e-15

# Input Thermal parameter
const α = 6.0e-4
const c = 1.0
const κ_mat = 1.0
const ρ = 0.0

function ElasFourthOrderConstTensor(E ,ν , PlanarState)
    # 1 for Plane Stress and 2 Plane Strain Condition
    if PlanarState == 1
        C1111 = E /(1 -ν *ν )
        C1122 = (ν *E ) /(1 -ν *ν )
        C1112 = 0.0
        C2222 = E /(1 -ν *ν )
        C2212 = 0.0
        C1212 = E /(2*(1+ν ) )
    elseif PlanarState == 2
        C1111 = (E *(1 -ν *ν ) ) /((1+ν ) *(1 -ν -2*ν *ν ) )
        C1122 = (ν *E ) /(1 -ν -2*ν *ν )
        C1112 = 0.0
        C2222 = (E *(1 -ν ) ) /(1 -ν -2*ν *ν )
        C2212 = 0.0
        C1212 = E /(2*(1+ν ) )
    end
    C_ten = SymFourthOrderTensorValue(C1111 , C1112 , C1122 , C1112 ,
    C1212 , C2212 , C1122 , C2212 , C2222)
    return C_ten
end

const C_mat = ElasFourthOrderConstTensor(E_mat,ν_mat,1)

κGradTemp(∇,s_in) = (s_in^2 + η)*κ_mat*∇

σ_elas(εElas) = C_mat ⊙ εElas

function σ_elasMod(ε, ε_in, s_in,T,T_in)
    
    εElas_in = ε_in - α*(T_in-T0)*I2
    εElas = ε - α*(T-T0)*I2
    
    if tr(εElas_in)  >= 0
        σ = (s_in^2 + η)*σ_elas(εElas)
    elseif tr(εElas_in) < 0
        σ = (s_in^2 + η) *I4_dev ⊙ σ_elas(εElas) + I4_vol⊙ σ_elas(εElas) 
    end  
    return σ
end

function σ_totMod(ε, ε_in,s_in,T_in)
    
    εElas_in = ε_in - α*(T_in-T0)*I2
    εElasTot = ε
    
    if tr(εElas_in)  >= 0
        σT = (s_in^2 + η)*σ_elas(εElasTot)
    elseif tr(εElas_in) < 0
        σT = (s_in^2 + η) *I4_dev ⊙ σ_elas(εElasTot) + I4_vol⊙ σ_elas(εElasTot) 
    end  
    return σT
end

function σ_totthMod(ε_in,s_in,T,T_in)
    
    εElas_in = ε_in - α*(T_in-T0)*I2
    εElasTotth = -α*T*I2
    
    if tr(εElas_in)  >= 0
        σT = (s_in^2 + η)*σ_elas(εElasTotth)
    elseif tr(εElas_in) < 0
        σT = (s_in^2 + η) *I4_dev ⊙ σ_elas(εElasTotth) + I4_vol⊙ σ_elas(εElasTotth) 
    end  

    return σT
end

function σ_thermMod(ε_in,s_in,T_in)
    
    εElas_in = ε_in - α*(T_in-T0)*I2
    εElasTher = α*(T0)*I2
    
    if tr(εElas_in)  >= 0
        σF = (s_in^2 + η)*σ_elas(εElasTher)
    elseif tr(εElas_in) < 0
        σF = (s_in^2 + η)*I4_dev ⊙ σ_elas(εElasTher) + I4_vol⊙ σ_elas(εElasTher) 
    end  
    return σF
end

function ψPos(ε_in,T_in)
    εElas_in = ε_in - α*(T_in-T0)*I2
    if tr(εElas_in)  >= 0
        ψPlus = 0.5*((εElas_in) ⊙ σ_elas(εElas_in))             
    elseif tr(εElas_in)  < 0
        ψPlus = 0.5*((I4_dev ⊙ σ_elas(εElas_in)) ⊙ (I4_dev ⊙ (εElas_in))) 
    end
    return ψPlus
end

function new_EnergyState(ψPlusPrev_in,ψhPos_in)
    ψPlus_in = ψhPos_in
    if ψPlus_in >= ψPlusPrev_in
        ψPlus_out = ψPlus_in
    else
        ψPlus_out = ψPlusPrev_in
    end
    true, ψPlus_out
end

function project(q,model,dΩ,order)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = FESpace(model,reffe,conformity=:L2)
    a(u,v) = ∫( u*v )*dΩ
    l(v) = ∫( v*q )*dΩ
    op = AffineFEOperator(a,l,V,V)
    qh = Gridap.solve(op)
    qh
end


order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

LoadTagId = get_tag_from_name(labels,"TopEdge")
Γ_Load = BoundaryTriangulation(model,tags = LoadTagId)
dΓ_Load = Measure(Γ_Load,degree)
n_Γ_Load = get_normal_vector(Γ_Load)


reffe_PF = ReferenceFE(lagrangian,Float64,order)
V0_PF = TestFESpace(model,reffe_PF;
  conformity=:H1)
U_PF = TrialFESpace(V0_PF)

reffe_Disp = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
        V0_Disp = TestFESpace(model,reffe_Disp;
          conformity=:H1,
          dirichlet_tags=["BottomEdge","TopEdge","LeftEdge","RightEdge"],
          dirichlet_masks=[(false,true),(false,true),(true,false),(true,false)])
uh = zero(V0_Disp)

reffe_Temp = ReferenceFE(lagrangian,Float64,order)
V0_Temp = FESpace(model,reffe_Temp;
  conformity=:H1,
  dirichlet_tags=["BottomEdge","TopEdge","LeftEdge","RightEdge"])

V0 = MultiFieldFESpace([V0_Disp,V0_Temp])

function  stepPhaseField(uh_in,ψPlusPrev_in)
        
    a_PF(s,ϕ) = ∫( Gc*lsp*∇(ϕ)⋅ ∇(s) + 2*ψPlusPrev_in*s*ϕ  + (Gc/lsp)*s*ϕ )*dΩ
    b_PF(ϕ) = ∫( (Gc/lsp)*ϕ )*dΩ
    op_PF = AffineFEOperator(a_PF,b_PF,U_PF,V0_PF)
    sh_out = Gridap.solve(op_PF)           
    
    return sh_out
    
end

 function stepDispTemp(uh_in,sh_in,T_in,vApp,TApp,delt)
    uApp1(x) = VectorValue(0.0,0.0)
    uApp2(x) = VectorValue(0.0,vApp)
    uApp3(x) = VectorValue(0.0,0.0)
    uApp4(x) = VectorValue(0.0,0.0)
    U_Disp = TrialFESpace(V0_Disp,[uApp1,uApp2,uApp3,uApp4])
    Tapp1(x) = 0.0
    Tapp2(x) = 0.0
    Tapp3(x) = 0.0
    Tapp4(x) = 0.0
    Tg = TrialFESpace(V0_Temp,[Tapp1,Tapp2,Tapp3,Tapp4])
    U = MultiFieldFESpace([U_Disp,Tg])
    a((u,T),(v,w)) = ∫( (ε(v) ⊙ (σ_totMod∘(ε(u),ε(uh_in),sh_in,T_in) + σ_totthMod∘(ε(uh_in),sh_in,T,T_in))) + ∇(w)⋅(κGradTemp∘ (∇(T),sh_in)) + ((ρ*c*T*w)/delt))*dΩ
    b((v,w)) = ∫(((ρ*c*T_in*w)/delt) - (ε(v) ⊙ (σ_thermMod∘(ε(uh_in),sh_in,T_in))))*dΩ
    op = AffineFEOperator(a,b,U,V0)
    uhTh = Gridap.solve(op)                
    uh_out,Th_out =  uhTh
    
    return uh_out,Th_out
end

t = 0.0
innerMax = 10
count = 0
tol = 1e-8

Load = Float64[]
Displacement = Float64[]

push!(Load, 0.0)
push!(Displacement, 0.0)

ψPlusPrev = CellState(0.0,dΩ) 
sPrev = CellState(1.0,dΩ)
sh = project(sPrev,model,dΩ,order)
ThPrev = CellState(T0,dΩ)
Th = project(ThPrev,model,dΩ,order)
while t .< tMax 
    count = count .+ 1      
    t = t + delt
    vApp = AppVel*t    
    TApp = TAppVec[count]

    print("\n Entering time step$count :", float(t))
    
   for inner = 1:innerMax   
        
        ψhPlusPrev = project(ψPlusPrev,model,dΩ,order)
        RelErr = abs(sum(∫( Gc*lsp*∇(sh)⋅ ∇(sh) + 2*ψhPlusPrev*sh*sh  + (Gc/lsp)*sh*sh)*dΩ - ∫( (Gc/lsp)*sh)*dΩ))/abs(sum(∫( (Gc/lsp)*sh)*dΩ))
        print("\n Relative error$count = ",float(RelErr))
        sh = stepPhaseField(uh,ψhPlusPrev) 
        uh,Th = stepDispTemp(uh,sh,Th,vApp,TApp,delt)

        ψhPos_in = ψPos∘(ε(uh),Th)   
        
        update_state!(new_EnergyState,ψPlusPrev,ψhPos_in)
  
        if RelErr < tol
            break 
        end      
    end
    
    Node_Force = sum(∫( n_Γ_Load ⋅ (σ_elasMod∘(ε(uh),ε(uh),sh,Th,Th)) ) *dΓ_Load) 
    
    push!(Load, Node_Force[2])     
    push!(Displacement, vApp)           
    if mod(count,5) == 0
         writevtk(Ω,"results_PhaseFieldThermoElastic$count",cellfields=
        ["uh"=>uh,"s"=>sh ,"epsi"=>ε(uh),"T"=>Th])
    end

end