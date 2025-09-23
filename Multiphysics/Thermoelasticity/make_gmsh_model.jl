
using GridapGmsh.gmsh
using GridapGmsh

gmsh.initialize()

mesh_size = 0.01
gmsh.model.geo.addPoint(0.1, 0., 0., mesh_size, 1)
gmsh.model.geo.addPoint(1.0, 0., 0., mesh_size, 2)
gmsh.model.geo.addPoint(1.0, 1.0, 0., mesh_size, 3)
gmsh.model.geo.addPoint(0., 1.0, 0., mesh_size, 4)
gmsh.model.geo.addPoint(0., 0.1, 0., mesh_size, 5)

gmsh.model.geo.addPoint(0., 0., 0., 6)
gmsh.model.geo.addCircleArc(5, 6, 1, 5)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)

gmsh.model.geo.addCurveLoop([1,2,3,4,5], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.addPhysicalGroup(2, [1], 1)
gmsh.model.addPhysicalGroup(1, [1], 1)
gmsh.model.addPhysicalGroup(1, [2] ,2)
gmsh.model.addPhysicalGroup(1, [3], 3)
gmsh.model.addPhysicalGroup(1, [4], 4)
gmsh.model.addPhysicalGroup(1, [5], 5)


gmsh.model.setPhysicalName(2, 1, "Domain")
gmsh.model.setPhysicalName(1, 1, "BottomEdge")
gmsh.model.setPhysicalName(1, 2, "RightEdge")
gmsh.model.setPhysicalName(1, 3, "TopEdge")
gmsh.model.setPhysicalName(1, 4, "LeftEdge")
gmsh.model.setPhysicalName(1, 5, "circArc")

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("mesh_TransientThermoElsticity.msh")
gmsh.finalize()

model = GmshDiscreteModel("mesh_TransientThermoElsticity.msh")
# writevtk(model, "mesh_TransientThermoElsticity")