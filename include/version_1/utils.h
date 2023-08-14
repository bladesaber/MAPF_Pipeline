#ifndef MAPF_PIPELINE_UTILS_H
#define MAPF_PIPELINE_UTILS_H

#include "iostream"
#include "vector"

template<typename T>
void printPointer(T& a, std::string tag){
    std::cout << tag << &a << std::endl;
}

#endif //MAPF_PIPELINE_UTILS_H