#include "constrainTable.h"

void ConstraintTable::insert2CT(double x, double y, double z, double radius){
    x_round = roundInterval(x, conflict_precision);
    y_round = roundInterval(y, conflict_precision);
    z_round = roundInterval(z, conflict_precision);

    // std::cout << "x_round:" << x_round << " y_round" << y_round << " z_round:" << z_round << std::endl;

    constrainTree->insertNode(0, x_round, y_round, z_round, radius, 0.0, 0.0);
    // ct.emplace_back(std::make_tuple(x_round, y_round, z_round, radius));
}

void ConstraintTable::insert2CT(ConstrainType constrain){
    double x, y, z, radius;
    std::tie(x, y, z, radius) = constrain;
    insert2CT(x, y ,z, radius);
}

bool ConstraintTable::isConstrained(double x, double y, double z, double radius){
    if (constrainTree->getTreeCount() == 0){
        return false;
    }

    KDTree_XYZRA_Res res;
    constrainTree->nearest(x, y, z, res);

    double dist = norm2_distance(x, y, z, res.x, res.y, res.z);
    if (dist <= radius + res.data->radius + eplision){
        return true;
    }
    return false;
}

bool ConstraintTable::isConstrained(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double radius
){
    double line_dist = norm2_distance(lineStart_x, lineStart_y, lineStart_z, lineEnd_x, lineEnd_y, lineEnd_z);
    double search_radius = std::sqrt(std::pow(line_dist/2.0, 2.0) + std::pow(radius * 1.05, 2.0));

    std::vector<KDTree_XYZRA_Res*> resList;
    bool isConstrain = false;

    constrainTree->nearest_range(lineStart_x, lineStart_y, lineStart_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {   
        if ( isConstrain == false ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );
            if (dist < radius + res->data->radius + eplision ){
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
        if ( isConstrain == false ){
            double dist = point2LineSegmentDistance(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                res->x, res->y, res->z
            );
            if ( dist < radius + res->data->radius + eplision ){
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
    double search_radius = std::sqrt(std::pow(line_dist/2.0, 2.0) + std::pow(radius * 1.05, 2.0));

    std::vector<KDTree_XYZRA_Res*> resList;
    bool isConstrain = false;

    constrainTree->nearest_range(lineStart_x, lineStart_y, lineStart_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {   
        if ( isConstrain == false ){
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
        return true;
    }
    
    resList.clear();
    constrainTree->nearest_range(lineEnd_x, lineEnd_y, lineEnd_z, search_radius, resList);
    for (KDTree_XYZRA_Res* res : resList)
    {
        if ( isConstrain == false ){
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
    
    return isConstrain;
}
