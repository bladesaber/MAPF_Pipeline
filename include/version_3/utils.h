#ifndef MAPF_PIPELINE_UTILS_H
#define MAPF_PIPELINE_UTILS_H

#include "common.h"
#include "kdtree.h"

double fmodr( double x, double y);

double mod2pi(double theta);

double mod2singlePi(double theta);

double rad2degree(double theta);

std::tuple<double, double, double> polar3D_to_vec3D(double alpha, double beta, double length);

std::tuple<double, double> vec3D_to_polar3D(double vec_x, double vec_y, double vec_z);

double point2LineDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
);

double norm2_distance(
    double x0, double y0, double z0,
    double x1, double y1, double z1
);

double point2LineSegmentDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
);

struct KDTreeWrapper{
    kdtree* tree;

    KDTreeWrapper(){
        this->tree = kd_create(3);
    }
    ~KDTreeWrapper(){
        this->free();
    }

    void free(){
        kd_free(this->tree);
    }

    void insertPoint(double x, double y, double z){
        kd_insert3(this->tree, x, y, z, 0);
    }

    void insertPath(DetailPath& path){
        double x, y, z;
        for (size_t i = 0; i < path.size(); i++)
        {
            std::tie(x, y, z) = path[i];
            kd_insert3(tree, x, y, z, 0);
        }
    }

    std::tuple<double, double, double> nearest(double x, double y, double z){
        double pos[3];
        pos[0] = x;
    	pos[1] = y;
	    pos[2] = z;
        kdres* res = kd_nearest(this->tree, pos);

        // // char *pch; data
        // while(!kd_res_end(res)){
        //   
        //     // get the data and position of the current result item
        //     // pch = (char*)kd_res_item(res, pos);
        //     // No data here
        //     kd_res_item(res, pos);
        //
        //     /* print out the retrieved data */
        //     // printf( "node at (%.3f, %.3f, %.3f) \n",  pos[0], pos[1], pos[2]);
        //
        //     /* go to the next entry */
        //     kd_res_next(res);
        // }

        kd_res_item(res, pos);
        
        return std::make_tuple(pos[0], pos[1], pos[2]);
    }

};

#endif