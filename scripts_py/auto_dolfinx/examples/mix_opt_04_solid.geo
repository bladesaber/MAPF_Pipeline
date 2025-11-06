//+
SetFactory("OpenCASCADE");

Mesh.MeshSizeMax = 0.05;
Mesh.MeshSizeMin = 0.05;


//+
Rectangle(2) = {0, 0, 0, 5, 5, 0};
//+
Physical Surface("volume", 5) = {2};
//+
Physical Curve("wall_fix", 6) = {3, 1};
//+
Physical Curve("wall_free", 7) = {4, 2};
