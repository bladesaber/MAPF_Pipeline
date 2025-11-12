### Package Included
   - conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista
   - python == 3.13.9


### TODO
   1. CBS路径搜索效率取决于障碍点的选择，CBS会沿着解空间的shape搜索，这近似单纯形，参考mapf.info的LNS方法更好,LNS方法也有利于改变局部拓扑结构
   2. 为路径搜索添加软障碍与组障碍
   3. 直接采用拓扑优化代替路径搜索
