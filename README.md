# MAPF_Pipeline

1. Package Included:
   1. Eigen3==3.3.4
   2. Fenicsx (Used For Finite Element Analysis)
   3. Gmsh & Netgen (Used For Mesh)
  

2. Problem Statement:
   1. dolfinx/dolfin is conflict with conda/g2o/pybind, they must be installed in different environments. Please do not install dolfin, it will destroy the whole conda/g2o environment


3. Question:
   1. CBS路径搜索效率取决于conflict的选择，CBS会沿着解空间的shape搜索，这近似单纯形，参考mapf.info的LNS方法更好
   2. 路径平滑中曲率优化效果不足够好。B样条的导数表达式不直观。更自由的参数化路径更好，但需要设计合适的函数族，目前缺乏精力处理
   3. 无法避免通路路径的自交叉现象,可以尝试检测自相交（当路径点A半径范围内的存在点不与A在同一连通域称为自相交）现象来抑制
   4. 先完成 Volumetric B-spline 与 shape optimisation 的 dolfinx 实现