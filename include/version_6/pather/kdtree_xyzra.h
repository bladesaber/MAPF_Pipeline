#ifndef MAPF_PIPELINE_OBS_KDTREE_H
#define MAPF_PIPELINE_OBS_KDTREE_H

#include "common.h"
#include "utils.h"
#include <assert.h>
#include "kdtree.h"

namespace PathNameSpace{

struct KDTree_XYZRA_Data{
    size_t idx;
    double radius;
    double alpha, theta;

    KDTree_XYZRA_Data(
        size_t idx, double radius, double theta, double alpha
    ):idx(idx), radius(radius), alpha(alpha), theta(theta){};
};

struct KDTree_XYZRA_Res{
    double x, y, z;
    KDTree_XYZRA_Data* data;

    KDTree_XYZRA_Res(){};
    KDTree_XYZRA_Res(double x, double y, double z, KDTree_XYZRA_Data* data):x(x), y(y), z(z), data(data){};
};

class KDTree_XYZRA
{
public:
    KDTree_XYZRA(){
        tree = kd_create(3);
    };
    ~KDTree_XYZRA(){
        release();
    };

    void insertNode(size_t idx, double x, double y, double z, double radius, double alpha, double theta);
    void nearest(double x, double y, double z, KDTree_XYZRA_Res& res);
    void nearest_range(double x, double y, double z, double bound, std::vector<KDTree_XYZRA_Res*>& resList);

    int getTreeCount(){
        return dataStore.size();
    };

private:
    kdtree* tree;
    std::vector<KDTree_XYZRA_Data*> dataStore;

    void release(){
        kd_free(tree);
        for (size_t i = 0; i < dataStore.size(); i++)
        {
            delete dataStore[i];
        }
        dataStore.clear();        
    };

};

}

#endif