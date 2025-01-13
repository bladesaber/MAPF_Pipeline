# MAPF_Pipeline

1. Package Included:
   1. Eigen3==3.3.4
   2. Fenicsx (Used For Finite Element Analysis)
   3. Gmsh & Netgen (Used For Mesh)
  

2. Problem Statement:
   1. dolfinx/dolfin is conflict with conda/g2o/pybind, they must be installed in different environments. Please do not install dolfin, it will destroy the whole conda/g2o environment


3. Question:
   1. CBS路径搜索效率取决于障碍点的选择，CBS会沿着解空间的shape搜索，这近似单纯形，参考mapf.info的LNS方法更好,LNS方法也有利于改变局部拓扑结构
   2. 完成 Volumetric B-spline 与 shape optimisation 的 dolfinx 实现
   3. 仿真自动化流程
   4. 目前的搜索结构注定后面的连接一定与前面相连，分区独立搜索再联合其实更好，更适用于分布式计算，做s3x3例子时很明显的显露出来
   5. Astar搜索时应该空阔时跨步更达，在密集环境（障碍物或其他管道附近）跨步更小
   6. 碰撞检测运行太慢,应该提前定义好图而不使用碰撞检测
   7. 膨胀方法必须处理自相交问题，膨胀方法无法生成异拓扑结构，需要其他辅助
   8. jump point search 取代 Astar
   9. 将Astar搜索的碰撞检测置换为限制图效率更高
   10. 组路径最短不是完备的
   11. 后处理:CAD服务器,STEP生成,STEP的FFD移动,STL的FFD移动,曲面生成
   12. 为路径搜索在进/出口之外添加中继点概念
   13. 阀芯使用拓扑优化实现自动化设计
   14. 为路径搜索添加软障碍与组障碍