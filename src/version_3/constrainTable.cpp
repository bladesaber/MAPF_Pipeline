#include "constrainTable.h"

void ConstraintTable::insert2CT(int loc, double radius){
    this->ct[loc] = radius;
}

bool ConstraintTable::isConstrained(int loc) const{
    const auto it = this->ct.find(loc);
    if (it == ct.end())
    {
        return false;
    }
    return true;
}

bool ConstraintTable::islineOnSight(Instance& instance, int parent_loc, int child_loc, double bound) const{

    // TODO 这里错了，不是点到线的距离，是点到线段的距离

    double lineStart_x = instance.getXCoordinate(parent_loc);
    double lineStart_y = instance.getYCoordinate(parent_loc);
    double lineStart_z = instance.getZCoordinate(parent_loc);

    double lineEnd_x = instance.getXCoordinate(child_loc);
    double lineEnd_y = instance.getYCoordinate(child_loc);
    double lineEnd_z = instance.getZCoordinate(child_loc);

    for (auto iter : this->ct)
    {
        int constrainLoc = iter.first;
        double point_x = instance.getXCoordinate(constrainLoc);
        double point_y = instance.getYCoordinate(constrainLoc);
        double point_z = instance.getZCoordinate(constrainLoc);

        double distance = point2LineSegmentDistance(
            lineStart_x, lineStart_y, lineStart_z,
            lineEnd_x, lineEnd_y, lineEnd_z,
            point_x, point_y, point_z
        );

        // ------ Debug
        std::cout << "(lineStart_x:" << lineStart_x << " lineStart_y:" << lineStart_y << " lineStart_z:" << lineStart_z << ")";
        std::cout << "(lineEnd_x:" << lineEnd_x << " lineEnd_y:" << lineEnd_y << " lineEnd_z:" << lineEnd_z << ")";
        std::cout << "(point_x:" << point_x << " point_y:" << point_y << " point_z:" << point_z << ") -> " << distance;
        std::cout << std::endl;
        // ---------------------------------

        if (distance < bound + iter.second)
        {
            return false;
        }
    }
    
    return true;
}
