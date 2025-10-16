SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 3, 0.5, 0};
//+
Extrude {{0, 0, 1}, {3, 1.0, 0}, Pi/2} {
  Curve{2}; 
}
//+
Rectangle(3) = {3.5, 1, 0, 0.5, 2, 0};
//+
Extrude {{0, 0, 1}, {3, 3, 0}, Pi/2} {
  Curve{10}; 
}
//+
Rectangle(5) = {0, 3.5, 0, 3, 0.5, 0};
//+
BooleanUnion{ Surface{3}; Delete; }{ Surface{4}; Delete; }
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{2}; Delete; }
//+
BooleanUnion{ Surface{5}; Delete; }{ Surface{3}; Delete; }
//+
BooleanUnion{ Surface{5}; Delete; }{ Surface{1}; Delete; }
//+
Physical Surface("vol", 34) = {5};
//+
Physical Curve("inlet", 35) = {24};
//+
Physical Curve("outlet", 36) = {33};
//+
Physical Curve("wall", 37) = {25, 26, 27, 28, 23, 22, 29, 30, 31, 32};

Mesh.MeshSizeMax = 0.1;
Mesh.MeshSizeMin = 0.05;
