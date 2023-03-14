//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_TEST_H
#define MAPF_PIPELINE_TEST_H

#include "map"

int add(int i, int j) {
    return i + j;
}

class Test
{
public:
    Test(){};
    ~Test(){};

    std::map<int, std::vector<int>> ct;
};


#endif //MAPF_PIPELINE_TEST_H
