//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Rectangle(2) = {-0.05, 0.1, 0, 0.05, 0.2, 0};
//+
Rectangle(3) = {1.0, 0.7, 0, 0.05, 0.2, 0};
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{2}; Delete; }
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{3}; Delete; }
//+
Physical Surface("volume", 21) = {1};
//+
Physical Curve("inlet", 22) = {18};
//+
Physical Curve("outlet", 23) = {10};
//+
Physical Curve("wall", 24) = {15, 12, 16, 13, 14, 20, 19, 17, 9, 11};

Mesh.MeshSizeMax = 0.02;
Mesh.MeshSizeMin = 0.02;
