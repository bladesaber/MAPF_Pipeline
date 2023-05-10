#include "smoother_random.h"

void RandomStep_Smoother::findGroupPairObs(
    size_ut groupIdx, double x, double y, double z, double radius, std::vector<ObsType>& obsList
){
    double bound, distance;
    std::vector<KDTreeRes*> resList;

    for (auto iter2 : groupMap)
    {
        if (groupIdx == iter2.second->groupIdx){
            continue;
        }

        bound = radius + iter2.second->path->max_radius;
        iter2.second->pathTree->nearest_range(x, y, z, bound, resList);

        for (size_t j = 0; j < resList.size(); j++)
        {
            double distance = norm2_distance(
                x, y, z,
                resList[j]->x, resList[j]->y, resList[j]->z
            );
            if (distance < resList[j]->data->radius + radius)
            {
                obsList.emplace_back(std::make_tuple(
                    resList[j]->x, resList[j]->y, resList[j]->z, resList[j]->data->radius, iter2.second->groupIdx
                ));
            }

            delete resList[j];
        }
        resList.clear();
    }
}

void RandomStep_Smoother::findStaticObs(double x, double y, double z, double radius, std::vector<ObsType>& obsList){
    std::vector<KDTreeRes*> resList;

    staticObs_tree->nearest_range(x, y, z, radius + staticObsRadius, resList);
    for (size_t i = 0; i < resList.size(); i++){
        obsList.emplace_back(std::make_tuple(
            resList[i]->x, resList[i]->y, resList[i]->z, staticObsRadius, 999
        ));

        delete resList[i];
    }
    resList.clear();
}

double RandomStep_Smoother::getSmoothessLoss(Vector3D& xim1, Vector3D& xi, Vector3D& xip1){
    return (xip1 - xi).sqlength() + (xi -xim1).sqlength();
}

double RandomStep_Smoother::getCurvatureLoss(Vector3D& xim1, Vector3D& xi, Vector3D& xip1, bool debug){
    Vector3D vec1 = xi - xim1;
    Vector3D vec2 = xip1 - xi;

    double cosT1 = vec1.dot(vec2) / (vec1.length() * vec2.length());
    double loss = 1.0 - cosT1;
    
    if (debug){
        std::cout << "Loss:" << loss << " Angel:" << acos(cosT1) / M_PI * 180.0 << std::endl;
    }

    return loss;
}

double RandomStep_Smoother::getObscaleLoss(Vector3D& x, Vector3D& y, double bound){
    return std::pow((x - y).length() - bound * 1.2, 2.0);
}

double RandomStep_Smoother::getNodeLoss(
    GroupPath* groupPath, size_ut xi_nodeIdx, Vector3D& xi,
    std::vector<ObsType>& groupPairObsList, bool debug
){
    double smoothLoss = 0.0;
    double curvatureLoss = 0.0;
    double groupPairLoss = 0.0;

    Vector3D xim1, xip1;

    GroupPathNode* xim1_node;
    GroupPathNode* xi_node;
    GroupPathNode* xip1_node;

    xi_node = groupPath->nodeMap[xi_nodeIdx];
    
    for (size_ut pathIdx : xi_node->pathIdx_set)
    {
        xim1_node = groupPath->nodeMap[xi_node->parentIdxsMap[pathIdx]];
        xip1_node = groupPath->nodeMap[xi_node->childIdxsMap[pathIdx]];

        xim1 = Vector3D(xim1_node->x, xim1_node->y, xim1_node->z);
        xip1 = Vector3D(xip1_node->x, xip1_node->y, xip1_node->z);

        if (wSmoothness != 0.0){
            smoothLoss += getSmoothessLoss(xim1, xi, xip1) * wSmoothness;

            if (debug){
                std::cout << "PathIdx: " << pathIdx << " SmoothLoss:" << smoothLoss << std::endl;
            }        
        }

        if (wCurvature != 0.0){
            curvatureLoss += getCurvatureLoss(xim1, xi, xip1, debug) * wCurvature;

            if (debug){
                std::cout << "PathIdx: " << pathIdx << " CurvatureLoss:" << curvatureLoss << std::endl;
            }
        }

        if (wGoupPairObs != 0.0){
            double rhs_x, rhs_y, rhs_z, rhs_radius;
            size_ut rhs_groupIdx;
            double single_groupPairLoss = 0.0;

            for (size_t i = 0; i < groupPairObsList.size(); i++)
            {
                std::tie(rhs_x, rhs_y, rhs_z, rhs_radius, rhs_groupIdx) = groupPairObsList[i];
                
                Vector3D y = Vector3D(rhs_x, rhs_y, rhs_z);
                single_groupPairLoss += single_groupPairLoss + getObscaleLoss(xi, y, rhs_radius + xi_node->radius) * wGoupPairObs;
            }
            
            single_groupPairLoss = single_groupPairLoss / (double)(groupPairObsList.size() + 0.00001);
            groupPairLoss += single_groupPairLoss;

            if (debug){
                std::cout << "PathIdx: " << pathIdx << " groupPairLoss:" << single_groupPairLoss << std::endl;
            }        
        }
    }

    return (smoothLoss + curvatureLoss + groupPairLoss) / (double)xi_node->pathIdx_set.size();
}

void RandomStep_Smoother::updateGradient(){
    GroupSmoothInfo* groupInfo;
    GroupPath* groupPath;

    size_ut xi_nodeIdx;
    GroupPathNode* xi_node;

    std::vector<ObsType> groupPairObsList;
    Vector3D xi, xi_tem;
    double best_loss, step_loss;

    for (auto iter_map : groupMap)
    {
        groupInfo = iter_map.second;
        groupPath = groupInfo->path;

        for (auto iter_path : groupPath->nodeMap)
        {
            Vector3D best_step;
            xi_nodeIdx = iter_path.first;

            // The First And Last Point Are Fixed
            if (groupPath->fixedNodes.find(xi_nodeIdx) != groupPath->fixedNodes.end())
            {
                groupInfo->grads[xi_nodeIdx] = best_step;
                continue;
            }

            xi_node = groupPath->nodeMap[xi_nodeIdx];
            xi = Vector3D(xi_node->x, xi_node->y, xi_node->z);

            if (wGoupPairObs != 0.0){
                groupPairObsList.clear();
                findGroupPairObs(groupInfo->groupIdx, xi_node->x, xi_node->y, xi_node->z, xi_node->radius, groupPairObsList);
            }

            best_loss = getNodeLoss(groupPath, xi_nodeIdx, xi, groupPairObsList, false);

            for (Vector3D step: this->steps)
            {
                xi_tem = xi + step;
                if (! isValidPos(xi_tem.getX(), xi_tem.getY(), xi_tem.getZ()))
                {
                    continue;
                }

                step_loss = getNodeLoss(groupPath, xi_nodeIdx, xi_tem, groupPairObsList, false);

                if (step_loss < best_loss)
                {
                    best_loss = step_loss;
                    best_step = step;
                }                
            }

            groupInfo->grads[xi_nodeIdx] = best_step;
        }
    }
}

void RandomStep_Smoother::smoothPath(size_t updateTimes){
    GroupSmoothInfo* groupInfo;
    GroupPath* groupPath;

    size_t step = 0;

    while (step < updateTimes)
    {
        updateGradient();

        for (auto iter : groupMap){
            groupInfo = iter.second;
            groupPath = groupInfo->path;

            for (auto iter_path : groupPath->nodeMap){
                size_ut nodeIdx = iter_path.first;

                groupPath->nodeMap[nodeIdx]->updateGrad( groupInfo->grads[nodeIdx] );
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
            if (start_shift > 0){
                newPath.emplace_back(std::make_tuple(
                    x + xStart_shift + x_shift, 
                    y + yStart_shift + y_shift, 
                    z + zStart_shift + z_shift, 
                    0.0
                ));
            }

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

            if (end_shift > 0){
                newPath.emplace_back(std::make_tuple(
                    x + xEnd_shift + x_shift, 
                    y + yEnd_shift + y_shift, 
                    z + zEnd_shift + z_shift, 
                    length + start_shift + end_shift
                ));
            }

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
