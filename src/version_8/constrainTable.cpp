#include "constrainTable.h"

void ConstraintTable::insert2CT(double x, double y, double z, double radius){
    max_constrain_radius = std::max(radius, max_constrain_radius);
    constrainTree->insertNode(0, x, y, z, radius, 0.0, 0.0);
}

void ConstraintTable::insert2CT(ConstrainType constrain){
    double x, y, z, radius;
    std::tie(x, y, z, radius) = constrain;

    double x_round = roundInterval(x, conflict_precision);
    double y_round = roundInterval(y, conflict_precision);
    double z_round = roundInterval(z, conflict_precision);
    insert2CT(x_round, y_round ,z_round, radius);
}

bool ConstraintTable::isConstrained(double x, double y, double z, double radius){
    if (constrainTree->getTreeCount() == 0){
        return false;
    }

    KDTree_XYZRA_Res res;
    constrainTree->nearest(x, y, z, res);

    double dist = norm2_distance(x, y, z, res.x, res.y, res.z);
    if (dist <= radius + res.data->radius + eplision){
//         std::cout << "(x:" << x << " y:" << y << " z:" << z << ")";
//         std::cout << " -> (res_x:" << res.x << " res_y:" << res.y << " res_z:" << res.z << ") dist:" << dist;
//         std::cout << " radius:" << radius << " res_radius:" << res.data->radius << std::endl;
        return true;
    }
    return false;
}

bool ConstraintTable::isConstrained(
    double lineStart_x, double lineStart_y, double lineStart_z, double lineEnd_x, double lineEnd_y, double lineEnd_z, double radius
){
    double line_dist = norm2_distance(lineStart_x, lineStart_y, lineStart_z, lineEnd_x, lineEnd_y, lineEnd_z);
    double search_radius = std::sqrt(
        std::pow( line_dist/2.0, 2.0 ) + std::pow( (radius + max_constrain_radius) * 1.05, 2.0 )
    );

//     std::cout << "lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << std::endl;
//     std::cout << "lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << std::endl;
//     std::cout << "radius:" << radius << " max_constrain_radius:" << max_constrain_radius;
//     std::cout << " line_dist:" << line_dist << " search_radius:" << search_radius << std::endl;

//     bool sign0 = lineStart_x==32 && lineStart_y==27 && lineStart_z==55;
//     bool sign1 = lineEnd_x==32 && lineEnd_y==28 && lineEnd_z==55;
//     if (sign0 && sign1){
//         std::cout << "lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << std::endl;
//         std::cout << "lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << std::endl;
//         std::cout << "radius:" << radius << " max_radius:" << max_constrain_radius;
//         std::cout << " line_dist:" << line_dist << " search_radius:" << search_radius << std::endl;
//     }

    std::vector<KDTree_XYZRA_Res*> resList;
    bool isConstrain = false;

    constrainTree->nearest_range(lineStart_x, lineStart_y, lineStart_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {   
        if ( !isConstrain ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );

            if (dist < radius + res->data->radius + eplision){
                // std::cout << "near_x:" << res->x << " near_y:" << res->y << " near_z:" << res->z << std::endl;
                // std::cout << "point2LineSegmentDistance: " << dist << " obs_radius:" << res->data->radius << " bound:" << radius + res->data->radius << std::endl;
                isConstrain = true;
            }
        }        
        delete res;
    }

    if (isConstrain){
        return true;
    }
    
    resList.clear();
    constrainTree->nearest_range(lineEnd_x, lineEnd_y, lineEnd_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {
        if ( !isConstrain ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );
            if ( dist < radius + res->data->radius + eplision){
                isConstrain = true;
            }
        }
        delete res;
    }
    
    return isConstrain;
}

bool ConstraintTable::islineOnSight(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double radius
){
    double line_dist = norm2_distance(lineStart_x, lineStart_y, lineStart_z, lineEnd_x, lineEnd_y, lineEnd_z);
    double search_radius = std::sqrt(
        std::pow( line_dist/2.0, 2.0 ) + 
        std::pow( (radius + max_constrain_radius) * 1.05, 2.0 )
    );

    std::vector<KDTree_XYZRA_Res*> resList;
    bool isConstrain = false;

    constrainTree->nearest_range(lineStart_x, lineStart_y, lineStart_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {   
        if ( !isConstrain ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );
            if (dist < (radius + res->data->radius + eplision) ){
                isConstrain = true;
            }
        }        
        delete res;
    }

    if (isConstrain){
        return false;
    }
    
    resList.clear();
    constrainTree->nearest_range(lineEnd_x, lineEnd_y, lineEnd_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {
        if ( !isConstrain ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );
            if (dist < (radius + res->data->radius + eplision) ){
                isConstrain = true;
            }
        }
        delete res;
    }
    
    return !isConstrain;
}
