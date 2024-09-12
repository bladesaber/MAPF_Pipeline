SetFactory("OpenCASCADE");

Box(1) = {0, 0, 0, 1, 0.8, 0.4};
Cylinder(2) = {-0.1, 0.15, 0.2, 0.1, 0, 0, 0.1, 2*Pi};
Cylinder(3) = {-0.1, 0.65, 0.2, 0.1, 0, 0, 0.1, 2*Pi};
Cylinder(4) = {1, 0.65, 0.2, 0.1, 0, 0, 0.1, 2*Pi};
Cylinder(5) = {1, 0.15, 0.2, 0.1, 0, 0, 0.1, 2*Pi};
BooleanUnion{ Volume{1}; Delete; }{ Volume{3}; Volume{2}; Volume{5}; Volume{4}; Delete; }

Physical Volume("volume0", 25) = {1};

Physical Surface("inlet0", 26) = {9};
Physical Surface("inlet1", 27) = {10};
Physical Surface("outlet0", 28) = {14};
Physical Surface("outlet1", 29) = {13};
Physical Surface("boundary", 30) = {3, 4, 5, 2, 8, 1, 6, 7, 11, 12};

Mesh.MeshSizeMax = 0.04;
Mesh.MeshSizeMin = 0.04;

