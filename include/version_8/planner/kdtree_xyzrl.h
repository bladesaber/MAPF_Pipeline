#ifndef MAPF_PIPELINE_PATH_KDTREE_H
#define MAPF_PIPELINE_PATH_KDTREE_H

#include "common.h"
#include "utils.h"
#include <assert.h>
#include "kdtree.h"

namespace PlannerNameSpace{

struct KDTree_XYZRL_Data{
    size_t idx;
    double radius;
    double length;

    KDTree_XYZRL_Data(size_t idx, double radius, double length):idx(idx), radius(radius), length(length){};
};

struct KDTree_XYZRL_Res{
    double x, y, z;
    KDTree_XYZRL_Data* data;

    KDTree_XYZRL_Res(){};
    KDTree_XYZRL_Res(double x, double y, double z, KDTree_XYZRL_Data* data):x(x), y(y), z(z), data(data){};
};

class KDTree_XYZRL
{
public:
    KDTree_XYZRL(){
        tree = kd_create(3);
    };
    ~KDTree_XYZRL(){
        release();
    };

    void insertNode(size_t idx, double x, double y, double z, double radius, double length){
        KDTree_XYZRL_Data* m = new KDTree_XYZRL_Data(idx, radius, length);
        dataStore.emplace_back(m);
        kd_insert3(tree, x, y, z, m);
    }

    void nearest(double x, double y, double z, KDTree_XYZRL_Res& res){
        double pos[3] = {x, y, z};
        kdres* treeRes;
        treeRes = kd_nearest(tree, pos);

        KDTree_XYZRL_Data* treeData = (KDTree_XYZRL_Data*)kd_res_item(treeRes, pos);

        res.x = pos[0];
        res.y = pos[1];
        res.z = pos[2];
        res.data = treeData;

        kd_res_free(treeRes);
    }
    
    void nearest_range(double x, double y, double z, double bound, std::vector<KDTree_XYZRL_Res*>& resList){
        double pos[3] = {x, y, z};
        kdres* treeRes = kd_nearest_range(tree, pos, bound);

        // std::cout << "[DEBUG]: Found Result Size: " << kd_res_size(treeRes) << std::endl;

        KDTree_XYZRL_Data* data;
        while(!kd_res_end(treeRes)){
            data = (KDTree_XYZRL_Data*)kd_res_item(treeRes, pos);

            KDTree_XYZRL_Res* res = new KDTree_XYZRL_Res();
            res->x = pos[0];
            res->y = pos[1];
            res->z = pos[2];
            res->data = data;
            resList.emplace_back(res);

            kd_res_next(treeRes);
        }

        kd_res_free(treeRes);
    }

    int getTreeCount(){
        return dataStore.size();
    };

private:
    kdtree* tree;
    std::vector<KDTree_XYZRL_Data*> dataStore;

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