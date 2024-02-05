# MAPF_Pipeline

1. Package Version:
    1. Eigen3==3.3.4
   2. Netgen (Used For Mesh and Finite Element Analysis)
   3. Fenicsx (Used For Finite Element Analysis)
   4. Gmsh (Used For Mesh)

2. Problem Statement:
   1. dolfinx/dolfin is conflict with conda/g2o/pybind, they must be installed in different environments. Please do not install dolfin, it will destroy the whole conda/g2o environment