#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"
#include "utils.h"
#include "instance.h"
#include "kdtree_xyzra.h"

using namespace PathNameSpace;

class ConstraintTable{
public:
    ConstraintTable(){
        constrainTree = new KDTree_XYZRA();
    };
    ~ConstraintTable(){
        // ct.clear();
        delete constrainTree;
    };

    // 要设置成能被步长整除
    const double conflict_precision = 0.1;

    // 由于计算精度无法调整好，为保证不出现自循环，这里的约束要比CBS的 sampleDetailPath 与 CBSNode的 findAllAgentConflict 更为严格
    const double scale = 1.05;
    const double eplision = 0.0;
    double max_constrain_radius = 0.0;

    void insert2CT(double x, double y, double z, double radius);
    void insert2CT(ConstrainType constrain);

    bool isConstrained(double x, double y, double z, double radius);
    bool isConstrained(
        double lineStart_x, double lineStart_y, double lineStart_z,
        double lineEnd_x, double lineEnd_y, double lineEnd_z,
        double radius
    );

    bool islineOnSight(
        double lineStart_x, double lineStart_y, double lineStart_z,
        double lineEnd_x, double lineEnd_y, double lineEnd_z,
        double radius
    );

    int getTreeCount(){
        return constrainTree->getTreeCount();
    }

private:
    KDTree_XYZRA* constrainTree;
};

#endif