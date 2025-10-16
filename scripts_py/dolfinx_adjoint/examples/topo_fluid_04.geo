//+
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 2, 1, 0};
//+
Rectangle(2) = {2, 0, 0, 3, 4, 0};
//+
Rectangle(3) = {5, 3, 0, 2, 1, 0};
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{2}; Delete; }
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{3}; Delete; }
//+
Physical Surface("block_surface", 17) = {1};
//+
Physical Curve("inlet", 18) = {16};
//+
Physical Curve("outlet", 19) = {10};
//+
Physical Curve("wall", 20) = {15, 14, 13, 9, 12, 11};

Mesh.MeshSizeMax = 0.1;
Mesh.MeshSizeMin = 0.05;
