#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"
#include "utils.h"
#include "instance.h"
#include "kdtreeWrapper.h"

class ConstraintTable{
public:
    ConstraintTable(){};
    ~ConstraintTable(){
        ct.clear();
    };

    // 要设置成能被步长整除
    double conflict_precision = 0.1;

    void insert2CT(double x, double y, double z, double radius);
    void insert2CT(ConstrainType constrain);

    bool isConstrained(double x, double y, double z, double radius);
    bool isConstrained(Instance& instance, int parent_loc, int child_loc, double radius);

    bool islineOnSight(Instance& instance, int parent_loc, int child_loc, double bound);

private:
    KDTreeWrapper constrainTree;
    std::vector<std::tuple<double, double, double, double>> ct;

    // template params
    double x_round, y_round, z_round;
    double lineStart_x, lineStart_y, lineStart_z;
    double lineEnd_x, lineEnd_y, lineEnd_z;
};

#endif