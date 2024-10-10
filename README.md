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
   5. 现在回想起来，单变量似乎并非非凸形的，似乎之前我的判断有出错，似乎拓扑优化的路径构造并非不可实现
   6. 仿真自动化流程
   7. 目前的搜索结构注定后面的连接一定与前面相连，分区独立搜索再联合其实更好，更适用于分布式计算，做s3x3例子时很明显的显露出来
   8. Astar搜索时应该空阔时跨步更达，在密集环境（障碍物或其他管道附近）跨步更小
   9. 碰撞检测方法运行太慢
   10. 膨胀方法必须处理自相交问题，膨胀方法无法生成异拓扑结构，需要其他辅助
   11. Jump Point Search取代Astar
   12. LNS方法也有利于改变局部拓扑结构
   13. CBS的速度与选择障碍点相关，这是个大问题
   14. 将Astar搜索的碰撞检测置换为限制图效率更高
   15. 平滑优化的时间很长，分开两步（1）control path conflict （2）spline path conflict