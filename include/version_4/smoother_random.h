#ifndef MAPF_PIPELINE_SMOOTHER_RANDOM_H
#define MAPF_PIPELINE_SMOOTHER_RANDOM_H

#include "common.h"
#include "vector3d.h"
#include "utils.h"
#include "instance.h"
#include "kdtreeWrapper.h"
#include "groupPath.h"

typedef std::tuple<double, double, double, double, size_ut> ObsType;

class GroupSmoothInfo{
public:
    size_ut groupIdx;
    GroupPath* path;
    KDTreeWrapper* pathTree;

    std::map<size_ut, Vector3D> grads;

    GroupSmoothInfo(GroupPath* path): path(path)
    {   
        groupIdx = path->groupIdx;
        pathTree = new KDTreeWrapper();

        GroupPathNode* node;
        for (auto iter : path->nodeMap)
        {
            node = iter.second;
            pathTree->insertPathNode(node->nodeIdx, node->x, node->y, node->z, node->radius);
        }
    };
    ~GroupSmoothInfo(){
        release();
    };

    void release(){
        delete pathTree;
        grads.clear();
    }

};

class RandomStep_Smoother{
public:
    double wSmoothness = 0.0;
    double wGoupPairObs = 0.0;
    double wCurvature = 0.0;
    double wStaticObs = 0.0;

    double staticObsRadius = 0.1;
    double stepReso;
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;

    std::vector<Vector3D> steps;

    std::map<size_ut, GroupSmoothInfo*> groupMap;

    RandomStep_Smoother(
        double xmin, double xmax,
        double ymin, double ymax,
        double zmin, double zmax,
        double stepReso = 0.1
    ): xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), stepReso(stepReso)
    {
        steps = {
            Vector3D(stepReso, 0.0, 0.0),
            Vector3D(0.0, stepReso, 0.0),
            Vector3D(0.0, 0.0, stepReso),
            
            Vector3D(-stepReso, 0.0, 0.0),
            Vector3D(0.0, -stepReso, 0.0),
            Vector3D(0.0, 0.0, -stepReso),

            Vector3D(stepReso, stepReso, 0.0),
            Vector3D(stepReso, -stepReso, 0.0),
            Vector3D(-stepReso, stepReso, 0.0),
            Vector3D(-stepReso, -stepReso, 0.0),

            Vector3D(stepReso, 0.0, stepReso),
            Vector3D(stepReso, 0.0, -stepReso),
            Vector3D(-stepReso, 0.0, stepReso),
            Vector3D(-stepReso, 0.0, -stepReso),

            Vector3D(0.0, stepReso, stepReso),
            Vector3D(0.0, stepReso, -stepReso),
            Vector3D(0.0, -stepReso, stepReso),
            Vector3D(0.0, -stepReso, -stepReso),

            Vector3D(stepReso, stepReso, stepReso),
            Vector3D(stepReso, stepReso, -stepReso),
            Vector3D(stepReso, -stepReso, stepReso),
            Vector3D(-stepReso, stepReso, stepReso),
            Vector3D(-stepReso, -stepReso, stepReso),
            Vector3D(stepReso, -stepReso, -stepReso),
            Vector3D(-stepReso, stepReso, -stepReso),
            Vector3D(-stepReso, -stepReso, -stepReso)
        };

        staticObs_tree = new KDTreeWrapper();
    };
    ~RandomStep_Smoother(){
        release();
    }

    void findGroupPairObs(size_ut groupIdx, double x, double y, double z, double radius, std::vector<ObsType>& obsList);
    void findStaticObs(double x, double y, double z, double radius, std::vector<ObsType>& obsList);

    void updateGradient();
    void smoothPath(size_t updateTimes);

    double getSmoothessLoss(Vector3D& xim1, Vector3D& xi, Vector3D& xip1);
    double getCurvatureLoss(Vector3D& xim1, Vector3D& xi, Vector3D& xip1, bool debug=false);
    double getObscaleLoss(Vector3D& x, Vector3D& y, double bound);
    double getNodeLoss(
        GroupPath* groupPath, size_ut xi_nodeIdx, Vector3D& xi,
        std::vector<ObsType>& groupPairObsList, std::vector<ObsType>& staticObsList,
        bool debug=false
    );

    void addDetailPath(size_ut groupIdx, size_ut pathIdx, DetailPath& detailPath, double radius){
        auto iter = groupMap.find(groupIdx);
        if (iter == groupMap.end()){
            GroupPath* new_groupPath = new GroupPath(groupIdx);
            new_groupPath->insertPath(pathIdx, detailPath, radius);

            GroupSmoothInfo* new_groupInfo = new GroupSmoothInfo(new_groupPath);
            groupMap[new_groupInfo->groupIdx] = new_groupInfo;
        }
        else
        {
            groupMap[groupIdx]->path->insertPath(pathIdx, detailPath, radius);
        }
    };

    DetailPath paddingPath(
        DetailPath& detailPath, 
        std::tuple<double, double, double> startPadding,
        std::tuple<double, double, double> endPadding,
        double x_shift, double y_shift, double z_shift
    );

    DetailPath detailSamplePath(DetailPath& path, double stepLength);

    bool isValidPos(double x, double y, double z){
        // std::cout << "x:"<< x << " y:" << y << " z:" << z;
        // std::cout << " xmin:" << xmin << " xmax:" << xmax << " ymin:" << ymin << " ymax:" << ymax << " zmin:" << zmin << " zmax:" << zmax << std::endl;

        if (x < xmin || x > xmax - 1)
        {
            return false;
        }

        if (y < ymin || y > ymax - 1)
        {
            return false;
        }
        
        if (z < zmin || z > zmax - 1)
        {
            return false;
        }

        return true;
    };

    void insertStaticObs(double x, double y, double z, double radius){
        staticObs_tree->insertObs(x, y, z, radius);
    }

private:
    KDTreeWrapper* staticObs_tree;

    void release(){
        for (auto iter : groupMap)
        {
            delete iter.second;
        }
        groupMap.clear();
        delete staticObs_tree;
    }

};

#endif