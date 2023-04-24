#include "kdtreeWrapper.h"

void KDTreeWrapper::insertPoint3D(double x, double y, double z, KDTreeData* data){
    kd_insert3(tree, x, y, z, data);
}

void KDTreeWrapper::insertPath3D(const DetailPath& path, double radius){
    double x, y, z, length;
    for (size_t i = 0; i < path.size(); i++)
    {
        std::tie(x, y, z, length) = path[i];
        KDTreeData* m = new KDTreeData(radius, length);

        dataStore.emplace_back(m);
        insertPoint3D(x, y, z, m);
    }
}

void KDTreeWrapper::nearest(double x, double y, double z, KDTreeRes& res){
    double pos[3] = {x, y, z};
    kdres* treeRes;
    treeRes = kd_nearest(tree, pos);

    KDTreeData* treeData = (KDTreeData*)kd_res_item(treeRes, pos);

    res.x = pos[0];
    res.y = pos[1];
    res.z = pos[2];
    res.data = treeData;

    kd_res_free(treeRes);
}

void KDTreeWrapper::debug_insert(){
    for(int i=0; i < 5; i++ ) {
        KDTreeData* data = new KDTreeData(i * 10., 0. + i);
        // kd_insert3(tree, (double)i, (double)i, (double)i, data);
        insertPoint3D((double)i, (double)i, (double)i, data);
    }
}

void KDTreeWrapper::debug_search(){
    // KDTreeData* pch;
    // kdres* presults;
    //
    // presults = kd_nearest_range(tree, pt, 2.0);
    // printf("found %d results:\n", kd_res_size(presults));
    //
    // double pos[3];
    // while(!kd_res_end( presults)) {
    //     pch = (KDTreeData*)kd_res_item(presults, pos);
    //
    //     std::cout << "x:" << pos[0] << " y:" << pos[1] << " z:" << pos[2];
    //     std::cout << " r:" << pch->radius << " l:" << pch->length << std::endl; 
    //
    //     kd_res_next( presults );
    // }

    KDTreeRes res;
    nearest(1.0, 1.0, 1.0, res);
    std::cout << "x:" << res.x << " y:" << res.y << " z:" << res.z;
    std::cout << " radius:" << res.data->radius << " length:" << res.data->length << std::endl;

    nearest(2.0, 2.0, 2.0, res);
    std::cout << "x:" << res.x << " y:" << res.y << " z:" << res.z;
    std::cout << " radius:" << res.data->radius << " length:" << res.data->length << std::endl;

    nearest(3.0, 3.0, 3.0, res);
    std::cout << "x:" << res.x << " y:" << res.y << " z:" << res.z;
    std::cout << " radius:" << res.data->radius << " length:" << res.data->length << std::endl;
}