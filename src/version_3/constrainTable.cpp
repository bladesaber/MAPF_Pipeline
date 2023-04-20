#include "constrainTable.h"

void ConstraintTable::insert2CT(double x, double y, double z, double radius){
    KDTreeData* m = new KDTreeData(radius, 0.0);
    constrainTree.insertPoint3D(x, y, z, m);
}

void ConstraintTable::insert2CT(ConstrainType constrain){
    double x, y, z, radius;
    std::tie(x, y, z, radius) = constrain;
    insert2CT(x, y ,z, radius);
}

bool ConstraintTable::isConstrained(double x, double y, double z, double radius){
    KDTreeRes res;
    double dist;

    constrainTree.nearest(x, y, z, res);

    dist = norm2_distance(
        x, y, z,
        res.x, res.y, res.z
    );
    if (dist < radius + res.data->radius)
    {
        return true;
    }
    return true;
}

bool ConstraintTable::islineOnSight(Instance& instance, int parent_loc, int child_loc, double bound){
    double lineStart_x = instance.getXCoordinate(parent_loc);
    double lineStart_y = instance.getYCoordinate(parent_loc);
    double lineStart_z = instance.getZCoordinate(parent_loc);

    double lineEnd_x = instance.getXCoordinate(child_loc);
    double lineEnd_y = instance.getYCoordinate(child_loc);
    double lineEnd_z = instance.getZCoordinate(child_loc);

    double point_x, point_y, point_z, distance, radius;
    for (size_t i = 0; i < ct.size(); i++)
    {
        std::tie(point_x, point_y, point_z, radius) = ct[i];

        distance = point2LineSegmentDistance(
            lineStart_x, lineStart_y, lineStart_z,
            lineEnd_x, lineEnd_y, lineEnd_z,
            point_x, point_y, point_z
        );

        // ------ Debug
        // std::cout << "(lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << ")";
        // std::cout << "(lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << ")";
        // std::cout << "(point_x:" << point_x << " point_y:" << point_y << " point_z:" << point_z << ") -> " << distance;
        // std::cout << std::endl;
        // ---------------------------------

        if (distance < bound + radius)
        {
            return false;
        }
    }
    
    return true;
}
