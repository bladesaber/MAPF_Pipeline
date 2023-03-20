//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_TEST_H
#define MAPF_PIPELINE_TEST_H

#include <iostream>

#include <pybind11/stl.h>
#include "vector"
#include "map"
#include "list"
#include "string"

// #include "instance.h"
// #include "cbsNode.h"

int testPring(int i, int j) {
    return i + j;
}

void debugPring_vector(std::vector<int>& a){
    for (int i : a)
    {
        std::cout << i << "->";
    }
    std::cout << std::endl;
}

void debugPring_list(std::list<int>& a){
    for (int i : a)
    {
        std::cout << i << "->";
    }
    std::cout << std::endl;
}

void debugPring_map(std::map<std::string, int>& a){
    for (auto it : a)
    {
        std::cout << it.first << " : " << it.second << std::endl;
    }
}

void debugPring_pair(const std::pair<std::string, int>& a){
    std::cout << a.first << " : " << a.second << std::endl;
}

void debugPring_tuple(const std::tuple<std::string, int>& a){
    std::cout << std::get<0>(a) << " : " << std::get<1>(a) << std::endl;
}

void debugPring(){
    int x = 10;
    int y = 5;
    int z = 1;

    std::list<std::tuple<int, int, int>> candidates{
        std::tuple<int, int, int>(y,   x+1, z),
        std::tuple<int, int, int>(y,   x-1, z),
        std::tuple<int, int, int>(y+1, x,   z),
        std::tuple<int, int, int>(y-1, x,   z),
        std::tuple<int, int, int>(y,   x,   z+1),
        std::tuple<int, int, int>(y,   x,   z-1)
    };
    for (auto next : candidates){
        x = std::get<0>(next);
        y = std::get<1>(next);
        z = std::get<2>(next);
        std::cout << x << " | " << y << " | " << z << std::endl;
    }
}

// 针对自定义对象 pybind 会传导指针
// void debugTransformArg_Ownclass(CBSNode* node){
//     node->g_val = 100;
// }
// void debugTransformArg_Ownclass(CBSNode& node){
//     node.g_val = 100;
// }
// 针对转换对象， pybind会将python类型转为C++类型，转换过程必然使指针变更，因为这是个新的类型
// void debugTransformArg(std::vector<int>* a){
//     a->emplace_back(10);
// }

#endif //MAPF_PIPELINE_TEST_H
