//+
SetFactory("OpenCASCADE");

Box(1) = {0, 0, 0, 1, 1, 0.3};
//+
Cylinder(2) = {-0.05, 0.15, 0.15, 0.05, 0, 0, 0.1, 2*Pi};
//+
Cylinder(3) = {1.0, 0.85, 0.15, 0.05, 0, 0, 0.1, 2*Pi};
//+
BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Volume{3}; Delete; }
//+
Physical Volume("volume", 19) = {1};
//+
Physical Surface("inlet", 20) = {8};
//+
Physical Surface("outlet", 21) = {10};
//+
Physical Surface("wall", 22) = {5, 4, 3, 2, 1, 7, 9, 6};

Field[1] = Box;
Field[1].VIn = 0.01;
Field[1].VOut = 0.03;
Field[1].XMin = -1.0;
Field[1].XMax = -0.04;
Field[1].YMin = 0.0;
Field[1].YMax = 0.3;
Field[1].ZMin = 0.0;
Field[1].ZMax = 0.3;
Field[1].Thickness = 0.1;

Field[2] = Box;
Field[2].VIn = 0.01;
Field[2].VOut = 0.03;
Field[2].XMin = 1.04;
Field[2].XMax = 1.1;
Field[2].YMin = 0.5;
Field[2].YMax = 1.0;
Field[2].ZMin = 0.0;
Field[2].ZMax = 0.3;
Field[2].Thickness = 0.1;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;

