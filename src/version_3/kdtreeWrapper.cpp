#include "kdtreeWrapper.h"

void KDTreeWrapper::insertPoint3D(double x, double y, double z){
    // kd_insert3(this->tree, x, y, z, 0);
}

void KDTreeWrapper::insertPath3D(DetailPath& path){
    // double x, y, z;
    // for (size_t i = 0; i < path.size(); i++)
    // {
    //     std::tie(x, y, z) = path[i];
    //     kd_insert3(tree, x, y, z, 0);
    // }
}

std::tuple<double, double, double> KDTreeWrapper::nearest(double x, double y, double z){
    double pos[3] = {x, y, z};
    kdres* res = kd_nearest(tree, pos);

    kd_res_item(res, pos);
    kd_res_free(res);
        
    return std::make_tuple(pos[0], pos[1], pos[2]);
}

void KDTreeWrapper::debug(){
    kdtree* ptree;
    char* data = "abcde";
    char* pch;
    struct kdres *presults;
    double pos[3], dist;
    double pt[3] = { 0, 0, 1 };
    double radius = 10;

    int num_pts = 5;

    /* create a k-d tree for 3-dimensional points */
    ptree = kd_create(3);

    /* add some random nodes to the tree (assert nodes are successfully inserted) */
    for(int i=0; i<num_pts; i++ ) {
        kd_insert3(ptree, (double)i, (double)i, (double)i, &data[i]);
    }

    /* find points closest to the origin and within distance radius */
    presults = kd_nearest_range( ptree, pt, radius );

    /* print out all the points found in results */
    printf( "found %d results:\n", kd_res_size(presults) );

    while( !kd_res_end( presults ) ) {
        /* get the data and position of the current result item */
        pch = (char*)kd_res_item( presults, pos );

        std::cout << "x:" << pos[0] << " y:" << pos[1] << " z:" << pos[2] << " r:" << *pch << std::endl; 

        /* go to the next entry */
        kd_res_next( presults );
    }

    /* free our tree, results set, and other allocated memory */
    kd_res_free( presults );
    kd_free( ptree );
}
