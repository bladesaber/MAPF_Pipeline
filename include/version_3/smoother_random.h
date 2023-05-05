#ifndef MAPF_PIPELINE_SMOOTHER_RANDOM_H
#define MAPF_PIPELINE_SMOOTHER_RANDOM_H

#include "common.h"
#include "vector3d.h"
#include "utils.h"
#include "smoother.h"
#include "instance.h"

typedef std::tuple<double, double, double, double, size_ut> ObsType;

class RandomStep_Smoother{
public:
    double wSmoothness = 0.0;
    double wObstacle = 0.0;
    double wCurvature = 0.0;

    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;

    std::vector<Vector3D> steps;

    std::map<size_ut, AgentSmoothInfo*> agentMap;

    RandomStep_Smoother(
        double xmin, double xmax,
        double ymin, double ymax,
        double zmin, double zmax,
        double stepReso = 0.1
    ): xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax)
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
    };
    ~RandomStep_Smoother(){
        release();
    }

    void updateGradient();
    void smoothPath(size_t updateTimes);

    double getSmoothessLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2);
    double getCurvatureLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2, bool debug=false);
    double getObscaleLoss(Vector3D& x, Vector3D& y, double bound);

    void findAgentObs(size_ut agentIdx, double x, double y, double z, double radius, std::vector<ObsType>& obsList);
    double getWholeLoss(
        size_ut agentIdx, double radius, 
        Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2,
        std::vector<ObsType>& obsList, bool debug=false
    );

    void addAgentObj(size_ut agentIdx, double radius, DetailPath& detailPath);

    DetailPath paddingPath(
        DetailPath& detailPath, 
        std::tuple<double, double, double> startPadding,
        std::tuple<double, double, double> endPadding,
        double x_shift, double y_shift, double z_shift
    );
    DetailPath detailSamplePath(DetailPath& path, double stepLength);

    bool isValidPos(double x, double y, double z);

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

#endif