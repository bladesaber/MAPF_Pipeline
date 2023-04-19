#ifndef MAPF_PIPELINE_KDTREEWRAPPER_H
#define MAPF_PIPELINE_KDTREEWRAPPER_H

#include "common.h"
#include "kdtree.h"

class KDTreeWrapper{
public:
    KDTreeWrapper(){
        tree = kd_create(3);
    }
    ~KDTreeWrapper(){
        release();
    }

    void clear(){
        kd_clear(tree);
    }

    void insertPoint3D(double x, double y, double z);

    void insertPath3D(DetailPath& path);

    std::tuple<double, double, double> nearest(double x, double y, double z);

    void debug();
    
private:
    kdtree* tree;

    void release(){
        kd_free(this->tree);
    }
};

#endif