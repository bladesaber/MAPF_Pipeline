#include "kdtree_xyzra.h"

namespace SmootherNameSpace{

void KDTree_XYZRA::insertNode(size_t idx, double x, double y, double z, double radius, double alpha, double theta){
    KDTree_XYZRA_Data* m = new KDTree_XYZRA_Data(idx, radius, alpha, theta);
    dataStore.emplace_back(m);
    kd_insert3(tree, x, y, z, m);
}

void KDTree_XYZRA::nearest(double x, double y, double z, KDTree_XYZRA_Res& res){
    double pos[3] = {x, y, z};
    kdres* treeRes;
    treeRes = kd_nearest(tree, pos);

    KDTree_XYZRA_Data* treeData = (KDTree_XYZRA_Data*)kd_res_item(treeRes, pos);

    res.x = pos[0];
    res.y = pos[1];
    res.z = pos[2];
    res.data = treeData;

    kd_res_free(treeRes);
}

void KDTree_XYZRA::nearest_range(
    double x, double y, double z, double bound, std::vector<KDTree_XYZRA_Res*>& resList
){
    double pos[3] = {x, y, z};
    kdres* treeRes = kd_nearest_range(tree, pos, bound);

    // std::cout << "[DEBUG]: Found Result Size: " << kd_res_size(treeRes) << std::endl;

    KDTree_XYZRA_Data* data;
    while(!kd_res_end(treeRes)){
        data = (KDTree_XYZRA_Data*)kd_res_item(treeRes, pos);

        KDTree_XYZRA_Res* res = new KDTree_XYZRA_Res();
        res->x = pos[0];
        res->y = pos[1];
        res->z = pos[2];
        res->data = data;
        resList.emplace_back(res);

        kd_res_next(treeRes);
    }

    kd_res_free(treeRes);
}

}
