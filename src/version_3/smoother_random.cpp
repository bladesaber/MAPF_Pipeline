#include "smoother_random.h"

void RandomStep_Smoother::addAgentObj(size_ut agentIdx, double radius, DetailPath& detailPath){
    AgentSmoothInfo* agentInfo =  new AgentSmoothInfo(agentIdx, radius, detailPath);
    agentMap[agentIdx] = agentInfo;
}

double RandomStep_Smoother::getSmoothessLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2){
    return (xip1 - xi).sqlength() + (xi -xim1).sqlength();
}

double RandomStep_Smoother::getCurvatureLoss(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2, bool debug){
    Vector3D vec0 = xim1 - xim2;
    Vector3D vec1 = xi - xim1;
    Vector3D vec2 = xip1 - xi;
    Vector3D vec3 = xip2 - xip1;

    double cosT0 = vec0.dot(vec1) / (vec0.length() * vec1.length());
    double loss0 = 1.0 - cosT0;

    double cosT1 = vec1.dot(vec2) / (vec1.length() * vec2.length());
    double loss1 = 1.0 - cosT1;

    double cosT2 = vec2.dot(vec3) / (vec2.length() * vec3.length());
    double loss2 = 1.0 - cosT2;
    
    double loss = loss0 + loss1 + loss2;

    if (debug)
    {
        std::cout << "Loss:" << loss << " ->" << loss0 << " + " << loss1 << " + " << loss << std::endl;
        std::cout << "Angel:" << acos(cosT0) / M_PI * 180.0 << " + " << acos(cosT1) / M_PI * 180.0 << " + " << acos(cosT2) / M_PI * 180.0 << std::endl;
    }

    return loss;
}

double RandomStep_Smoother::getObscaleLoss(Vector3D& x, Vector3D& y, double bound){
    return std::pow((x - y).length() - bound * 1.2, 2.0);
}

void RandomStep_Smoother::findAgentObs(
    size_ut agentIdx, double x, double y, double z, double radius, std::vector<ObsType>& obsList
){
    double bound, distance;
    std::vector<KDTreeRes*> resList;

    for (auto iter2 : agentMap)
    {
        if (agentIdx == iter2.second->agentIdx){
            continue;
        }

        bound = radius + iter2.second->radius;
        iter2.second->pathTree->nearest_range(x, y, z, bound, resList);

        for (size_t j = 0; j < resList.size(); j++)
        {
            obsList.emplace_back(std::make_tuple(
                resList[j]->x, resList[j]->y, resList[j]->z, resList[j]->data->radius, iter2.second->agentIdx
            ));

            delete resList[j];
        }
        resList.clear();
    }
}


double RandomStep_Smoother::getWholeLoss(
    size_ut agentIdx, double radius,
    Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2,
    std::vector<ObsType>& obsList, bool debug
){
    double smoothLoss = 0.0;
    double curvatureLoss = 0.0;
    double obsLoss = 0.0;

    if (wSmoothness != 0.0){
        smoothLoss = getSmoothessLoss(xim2, xim1, xi, xip1, xip2) * wSmoothness;

        if (debug){
            std::cout << "SmoothLoss:" << smoothLoss << std::endl;
        }        
    }

    if (wCurvature != 0.0){
        curvatureLoss = getCurvatureLoss(xim2, xim1, xi, xip1, xip2, debug) * wCurvature;

        if (debug){
            std::cout << "CurvatureLoss:" << curvatureLoss << std::endl;
        }
    }

    if (wObstacle != 0.0){
        double rhs_x, rhs_y, rhs_z, rhs_radius;
        size_ut rhs_agentIdx;
        for (size_t i = 0; i < obsList.size(); i++)
        {
            std::tie(rhs_x, rhs_y, rhs_z, rhs_radius, rhs_agentIdx) = obsList[i];
            
            Vector3D y = Vector3D(rhs_x, rhs_y, rhs_z);
            obsLoss = obsLoss + getObscaleLoss(xi, y, rhs_radius+radius) * wObstacle;
        }

        if (debug){
            std::cout << "ObsLoss:" << obsLoss << std::endl;
        }        
    }

    return smoothLoss + curvatureLoss + obsLoss;
}


void RandomStep_Smoother::updateGradient(){
    AgentSmoothInfo* agent;

    double x, y, z;
    std::vector<ObsType> obsList;

    Vector3D xi_tem;
    double best_loss, step_loss;

    for (auto iter : agentMap)
    {
        agent = iter.second;

        // The First And Last Point Are Fixed
        for (size_t i = 0; i < agent->pathXYZ.size(); i++)
        {
            // remember to init to zero vector
            Vector3D best_step;

            if (i <= 1 || i >= agent->pathXYZ.size()-2){
                agent->grads[i] = best_step;
                continue;
            }
            
            Vector3D xim2 = agent->pathXYZ[i-2];
            Vector3D xim1 = agent->pathXYZ[i-1];
            Vector3D xi = agent->pathXYZ[i];
            Vector3D xip1 = agent->pathXYZ[i+1];
            Vector3D xip2 = agent->pathXYZ[i+2];

            // std::cout << "AgentIdx:" << agent->agentIdx << " Point: " << i << std::endl;
            // std::cout << "  xim2" << xim2 << "xim1:" << xim1 << " xi:" << xi << " xip1:" << xip1 << " xip2:" << xip2 << std::endl;

            if (wObstacle != 0.0){
                x = xi.getX();
                y = xi.getY();
                z = xi.getZ();
                obsList.clear();

                findAgentObs(agent->agentIdx, x, y, z, agent->radius, obsList);
            }

            best_loss = getWholeLoss(
                agent->agentIdx, agent->radius,
                xim2, xim1, xi, xip1, xip2,
                obsList, false
            );
            // std::cout << "  OriginalLoss:" << best_loss << std::endl;

            for (Vector3D step: this->steps)
            {
                xi_tem = xi + step;
                if (! isValidPos(xi_tem.getX(), xi_tem.getY(), xi_tem.getZ()))
                {
                    continue;
                }

                step_loss = getWholeLoss(
                    agent->agentIdx, agent->radius,
                    xim2, xim1, xi_tem, xip1, xip2,
                    obsList, false
                );

                if (step_loss < best_loss)
                {
                    best_loss = step_loss;
                    best_step = step;
                }

                // // ---------------- Just For Debug
                // std::cout << "  xi_tem" << xi_tem << " stepLoss:" << step_loss << " BestLoss:" << best_loss << "BestStep:" << best_step << std::endl;
                // getWholeLoss(
                //     agent->agentIdx, agent->radius,
                //     xim2, xim1, xi_tem, xip1, xip2,
                //     obsList, true
                // );
                // // ------------------------------------------
                
            }

            // ---------------- Just For Debug
            // Vector3D test_xi = xi + best_step;
            // std::cout << "xi_tem" << test_xi << " stepLoss:" << step_loss << " BestLoss:" << best_loss << std::endl;
            // getWholeLoss(
            //     agent->agentIdx, agent->radius,
            //     xim2, xim1, test_xi, xip1, xip2,
            //     obsList, true
            // );
            // ------------------------------------------

            agent->grads[i] = best_step;
        }
    }
}

void RandomStep_Smoother::smoothPath(size_t updateTimes){
    AgentSmoothInfo* agent;
    Vector3D xyz;
    Vector3D grad;

    size_t step = 0;

    while (step < updateTimes)
    {
        updateGradient();

        for (auto iter : agentMap){
            agent = iter.second;

            for (size_t i = 0; i < agent->pathXYZ.size(); i++)
            {
                xyz = agent->pathXYZ[i];
                grad = agent->grads[i];

                agent->pathXYZ[i] = xyz + grad;
            }
        }

        step += 1;
    }
}

DetailPath RandomStep_Smoother::paddingPath(
    DetailPath& detailPath, 
    std::tuple<double, double, double> startPadding,
    std::tuple<double, double, double> endPadding,
    double x_shift, double y_shift, double z_shift
)
{
    double xStart_shift, yStart_shift, zStart_shift;
    double xEnd_shift, yEnd_shift, zEnd_shift;

    std::tie(xStart_shift, yStart_shift, zStart_shift) = startPadding;
    std::tie(xEnd_shift, yEnd_shift, zEnd_shift) = endPadding;

    DetailPath newPath;

    double x, y, z, length;
    double start_shift = sqrt(pow(xStart_shift, 2) + pow(yStart_shift, 2) + pow(zStart_shift, 2));
    double end_shift = sqrt(pow(xEnd_shift, 2) + pow(yEnd_shift, 2) + pow(zEnd_shift, 2));

    for (size_t i = 0; i < detailPath.size(); i++)
    {
        std::tie(x, y, z, length) = detailPath[i];

        if (i == 0)
        {
            newPath.emplace_back(std::make_tuple(
                x + xStart_shift + x_shift, 
                y + yStart_shift + y_shift, 
                z + zStart_shift + z_shift, 
                0.0
            ));

            newPath.emplace_back(std::make_tuple(
                x + x_shift, 
                y + y_shift, 
                z + z_shift, 
                length + start_shift
            ));

        }else if(i == detailPath.size()-1){
            newPath.emplace_back(std::make_tuple(
                x + x_shift, 
                y + y_shift, 
                z + z_shift, 
                length + start_shift
            ));

            newPath.emplace_back(std::make_tuple(
                x + xEnd_shift + x_shift, 
                y + yEnd_shift + y_shift, 
                z + zEnd_shift + z_shift, 
                length + start_shift + end_shift
            ));

        }else{

            newPath.emplace_back(std::make_tuple(
                x + x_shift, 
                y + y_shift, 
                z + z_shift, 
                length + start_shift
            ));
        }
    }

    return newPath;
}

DetailPath RandomStep_Smoother::detailSamplePath(DetailPath& path, double stepLength){
    DetailPath new_detail_path;

    double lastX, lastY, lastZ, lastLength;
    std::tie(lastX, lastY, lastZ, lastLength) = path[0];

    double curX, curY, curZ, curLength;
    double distance, real_stepLength;
    size_t num;
    double vecX, vecY, vecZ;
    double cur_length = 0.0;

    for (size_t i = 1; i < path.size(); i++){
        std::tie(curX, curY, curZ, curLength) = path[i];

        distance = norm2_distance(
            lastX, lastY, lastZ,
            curX, curY, curZ
        );

        if (distance < stepLength)
        {
            // cur_length += real_stepLength;
            // new_detail_path.emplace_back(std::make_tuple(
            //     curX, curY, curZ, cur_length
            // ));
            // lastX = curX;
            // lastY = curY;
            // lastZ = curZ;
            continue;
        }
        
        num = std::ceil(distance / stepLength);
        real_stepLength = distance / (double)num;
        vecX = (curX - lastX) / distance;
        vecY = (curY - lastY) / distance;
        vecZ = (curZ - lastZ) / distance;

        // for last point
        if (i == path.size() - 1){
            num += 1;
        }

        for (size_t j = 0; j < num; j++)
        {
            new_detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength),
                cur_length
            ));
            cur_length += real_stepLength;

        }

        lastX = curX;
        lastY = curY;
        lastZ = curZ;
    }

    return new_detail_path;
}

bool RandomStep_Smoother::isValidPos(double x, double y, double z){
    if (x < xmin || x > xmax)
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
}