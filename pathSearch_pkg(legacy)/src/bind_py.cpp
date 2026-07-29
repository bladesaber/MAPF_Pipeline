#include "bind_py.h"

PYBIND11_MODULE(MapfPkg, m)
{
    m.doc() = "MapfPkg";
    
    pybind_modules::bind_package(m);
}