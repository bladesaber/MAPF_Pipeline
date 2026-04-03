//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 0.5};
//+
Cylinder(2) = {-0.05, 0.2, 0.25, 0.05, 0, 0, 0.1, 2*Pi};
//+
Cylinder(3) = {1.0, 0.8, 0.3, 0.05, 0, 0, 0.1, 2*Pi};
//+
BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
BooleanUnion{ Volume{4}; Delete; }{ Volume{3}; Delete; }
//+
Physical Volume("volume", 19) = {1};
//+
Physical Surface("inlet", 20) = {8};
//+
Physical Surface("outlet", 21) = {10};
//+
Physical Surface("wall", 22) = {3, 5, 4, 2, 7, 1, 6, 9};

Field[1] = Box;
Field[1].VIn = 0.01;
Field[1].VOut = 0.035;
Field[1].XMin = -1.0;
Field[1].XMax = -0.04;
Field[1].YMin = 0.0;
Field[1].YMax = 1.0;
Field[1].ZMin = 0.0;
Field[1].ZMax = 0.5;
Field[1].Thickness = 0.1;

Field[2] = Box;
Field[2].VIn = 0.01;
Field[2].VOut = 0.035;
Field[2].XMin = 1.04;
Field[2].XMax = 1.1;
Field[2].YMin = 0.5;
Field[2].YMax = 1.0;
Field[2].ZMin = 0.0;
Field[2].ZMax = 0.5;
Field[2].Thickness = 0.1;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;
