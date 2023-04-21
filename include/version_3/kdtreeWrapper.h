#ifndef MAPF_PIPELINE_KDTREEWRAPPER_H
#define MAPF_PIPELINE_KDTREEWRAPPER_H

#include "common.h"
#include "utils.h"
#include <assert.h>

#include "kdtree.h"

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

    void insertPath3D(const DetailPath& path, double radius);

    void nearest(double x, double y, double z, KDTreeRes& res);

    void debug_insert();
    void debug_search();

private:
    kdtree* tree;
    std::vector<KDTreeData*> dataStore;

    void release(){
        kd_free(tree);
        for (size_t i = 0; i < dataStore.size(); i++)
        {
            delete dataStore[i];
        }
        dataStore.clear();        
    }
};

#endif