#ifndef MAPF_PIPELINE_SMOOTHER_H
#define MAPF_PIPELINE_SMOOTHER_H

#include "common.h"
#include "vector3d.h"
#include "utils.h"
#include "kdtreeWrapper.h"

/*
typedef std::vector<Vector3D> PathXYZ;

class AgentSmoothInfo{
public:
    size_ut agentIdx;
    
    PathXYZ pathXYZ;
    double radius;
    KDTreeWrapper* pathTree;

    std::vector<Vector3D> grads;

    AgentSmoothInfo(size_ut agentIdx, double radius, DetailPath& detailPath):
        agentIdx(agentIdx), radius(radius)
    {
        double x, y, z, length;
        for (size_t i = 0; i < detailPath.size(); i++)
        {
            std::tie(x, y, z, length) = detailPath[i];
            pathXYZ.emplace_back(Vector3D(x, y, z));
        }
        
        pathTree = new KDTreeWrapper();
        pathTree->insertPath3D(detailPath, radius);

        grads.resize(pathXYZ.size());
    }
    ~AgentSmoothInfo(){}

    void release(){
        delete pathTree;
        pathXYZ.clear();
        grads.clear();
    }

};
*/

/*
class Smoother{
public:
    double wSmoothness = 0.0;
    double wObstacle = 0.0;
    double wCurvature = 0.0;

    double alpha = 0.1;
    double gradMax = 3.0;

    std::map<size_ut, AgentSmoothInfo*> agentMap;

    Smoother(){};
    ~Smoother(){
        release();
    }

    void updateGradient();
    void smoothPath(size_t updateTimes);

    Vector3D getSmoothessGradent(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2);
    double getSmoothessLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2);

    Vector3D getCurvatureGradent(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2);
    Vector3D getGradientFirst(Vector3D& xim2, Vector3D& xim1, Vector3D& xi);
    Vector3D getGradientMid(Vector3D& xim1, Vector3D& xi, Vector3D& xip1);
    Vector3D getGradientLast(Vector3D& xi, Vector3D& xip1, Vector3D& xip2);
    double getCurvatureLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2, bool debug=false);
    
    Vector3D getObscaleGradent(Vector3D& x, Vector3D& y, double bound);

    void addAgentObj(size_ut agentIdx, double radius, DetailPath& detailPath);

    DetailPath paddingPath(
        DetailPath& detailPath, 
        std::tuple<double, double, double> startPadding,
        std::tuple<double, double, double> endPadding
    );

private:
    void release(){
        for (auto iter : agentMap)
        {
            iter.second->release();
            delete iter.second;
        }
        agentMap.clear();
    }

};
*/

#endif