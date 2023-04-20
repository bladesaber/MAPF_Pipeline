#ifndef MAPF_PIPELINE_KDTREEWRAPPER_H
#define MAPF_PIPELINE_KDTREEWRAPPER_H

#include "common.h"
#include "kdtree.h"
#include "utils.h"

struct KDTreeData{
    double radius;
    double length;

    KDTreeData(){};
    KDTreeData(double radius, double length):radius(radius), length(length){};
};

struct KDTreeRes{
    double x, y, z;
    KDTreeData* data;

    KDTreeRes(){};
    KDTreeRes(double x, double y, double z, KDTreeData* data):
        x(x), y(y), z(z), data(data){};

};

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

    void insertPoint3D(double x, double y, double z, KDTreeData* data);

    void insertPath3D(DetailPath& path, double radius);

    void nearest(double x, double y, double z, KDTreeRes& res);

    void debug_insert();
    void debug_search();

private:
    kdtree* tree;

    void release(){
        kd_free(tree);
    }

    // template params
    double x_round, y_round, z_round;
};

#endif