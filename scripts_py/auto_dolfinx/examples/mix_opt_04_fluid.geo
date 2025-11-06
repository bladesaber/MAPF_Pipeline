//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 2.25, 0, 5, 0.5, 0};
//+
Physical Surface("volume", 5) = {1};
//+
Physical Curve("inlet", 6) = {4};
//+
Physical Curve("outlet", 7) = {2};
//+
Physical Curve("wall", 8) = {3, 1};

Mesh.MeshSizeMax = 0.05;
Mesh.MeshSizeMin = 0.05;
