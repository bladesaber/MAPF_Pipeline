#include "constrainTable.h"

void ConstraintTable::insert2CT(double x, double y, double z, double radius){
    x_round = roundInterval(x, conflict_precision);
    y_round = roundInterval(y, conflict_precision);
    z_round = roundInterval(z, conflict_precision);

    // std::cout << "x_round:" << x_round << " y_round" << y_round << " z_round:" << z_round << std::endl;

    KDTreeData* m = new KDTreeData(radius, 0.0);
    constrainTree.insertPoint3D(x_round, y_round, z_round, m);

    ct.emplace_back(std::make_tuple(x_round, y_round, z_round, radius));
}

void ConstraintTable::insert2CT(ConstrainType constrain){
    double x, y, z, radius;
    std::tie(x, y, z, radius) = constrain;
    insert2CT(x, y ,z, radius);
}

bool ConstraintTable::isConstrained(double x, double y, double z, double radius){
    if (ct.size() == 0){
        return false;
    }

    // 这里不能再使用离散的判断方法，因为可能存在两个离散点都不产生约束，但两个离散点的中间段产生约束的情况
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
    return false;
}

bool ConstraintTable::isConstrained(Instance& instance, int parent_loc, int child_loc, double radius){
    lineStart_x = instance.getXCoordinate(parent_loc);
    lineStart_y = instance.getYCoordinate(parent_loc);
    lineStart_z = instance.getZCoordinate(parent_loc);

    lineEnd_x = instance.getXCoordinate(child_loc);
    lineEnd_y = instance.getYCoordinate(child_loc);
    lineEnd_z = instance.getZCoordinate(child_loc);

    double line_dist = norm2_distance(
        lineStart_x, lineStart_y, lineStart_z,
        lineEnd_x, lineEnd_y, lineEnd_z
    );

    double vec_x = (lineEnd_x - lineStart_x) / line_dist;
    double vec_y = (lineEnd_y - lineStart_y) / line_dist;
    double vec_z = (lineEnd_z - lineStart_z) / line_dist;

    size_t num = std::ceil(line_dist / sample_precision);
    double realStep = line_dist / num;

    // // ------ Just For Debug
    // std::cout << "(lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << ")";
    // std::cout << "(lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << ")";
    // std::cout << "Distance: " << line_dist << std::endl;
    // std::cout << "Vec_x: " << vec_x << " Vec_y" << vec_y << " Vec_z:" << vec_z << " num:" << num << std::endl;
    // // ----------------------------------------

    KDTreeRes res;
    double dist;
    double x, y, z;
    for (size_t i = 0; i < num + 1; i++)
    {
        x = lineStart_x + i * vec_x * realStep;
        y = lineStart_y + i * vec_y * realStep;
        z = lineStart_z + i * vec_z * realStep;
        constrainTree.nearest(x, y, z, res);

        dist = norm2_distance(
            x, y, z,
            res.x, res.y, res.z
        );

        // // ------ Debug
        // std::cout << "(x:" << x << " y:" << y << " z:" << z << ")";
        // std::cout << "(obs_x:" << res.x << " obs_y:" << res.y << " obs_z:" << res.z << ") -> " << dist;
        // std::cout << std::endl;
        // // ---------------------------------

        if ( dist < (radius + res.data->radius) * scale )
        {
            return true;
        }
    }
    return false;
}

bool ConstraintTable::islineOnSight(Instance& instance, int parent_loc, int child_loc, double bound){
    // TODO need to improve performance

    lineStart_x = instance.getXCoordinate(parent_loc);
    lineStart_y = instance.getYCoordinate(parent_loc);
    lineStart_z = instance.getZCoordinate(parent_loc);

    lineEnd_x = instance.getXCoordinate(child_loc);
    lineEnd_y = instance.getYCoordinate(child_loc);
    lineEnd_z = instance.getZCoordinate(child_loc);
    
    double point_x, point_y, point_z, distance, radius;
    for (size_t i = 0; i < ct.size(); i++)
    {
        std::tie(point_x, point_y, point_z, radius) = ct[i];

        distance = point2LineSegmentDistance(
            lineStart_x, lineStart_y, lineStart_z,
            lineEnd_x, lineEnd_y, lineEnd_z,
            point_x, point_y, point_z
        );

        // // ------ Debug
        // std::cout << "(lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << ")";
        // std::cout << "(lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << ")";
        // std::cout << "(point_x:" << point_x << " point_y:" << point_y << " point_z:" << point_z << ") -> " << distance;
        // std::cout << std::endl;
        // // ---------------------------------

        if (distance < (bound + radius) * scale )
        {
            return false;
        }
    }
    
    return true;
}
