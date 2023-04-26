#include "smoother.h"

void Smoother::addAgentObj(size_ut agentIdx, double radius, DetailPath& detailPath){
    AgentSmoothInfo* agentInfo =  new AgentSmoothInfo(agentIdx, radius, detailPath);
    agentMap[agentIdx] = agentInfo;
}

Vector3D Smoother::getObscaleGradent(Vector3D& x1, Vector3D& obs, double bound){
    Vector3D vec_dif = x1 - obs;
    double dist = vec_dif.length();

    if ( dist < bound)
    {
        Vector3D gradient = 2.0 * (-bound + dist) * vec_dif / dist;

        gradient.clamp(-gradMax, gradMax);

        return gradient;

    }else{
        Vector3D zeros;
        return zeros;
    }
}

Vector3D Smoother::getSmoothessGradent(Vector3D& xim2, Vector3D& xim1, Vector3D& xi, Vector3D& xip1, Vector3D& xip2){
    // Vector3D gradient = (xim1 * -2.0) + (xi * 4.0) - (xip1 * 2.0);
    // gradient.clamp(-gradMax, gradMax);

    Vector3D gradient = xim2 - 4 * xim1 + 6 * xi - 4 * xip1 + xip2;
    gradient.clamp(-gradMax, gradMax);

    return gradient;
}

Vector3D Smoother::getCurvatureGradent(Vector3D& xim1, Vector3D& xi, Vector3D& xip1){
    /* Reference By Hybrid Astar */

    Vector3D gradient;
    // the vectors between the nodes
    Vector3D Dxi = xi - xim1;
    Vector3D Dxip1 = xip1 - xi;
    // orthogonal complements vector
    Vector3D p1, p2;

    double absDxi = Dxi.length();
    double absDxip1 = Dxip1.length();

    if (absDxi > 0 && absDxip1 > 0) {
        double Dphi = std::acos(rangeClamp(Dxi.dot(Dxip1) / (absDxi * absDxip1), -1.0, 1.0));
        double kappa = Dphi / absDxi;

        if (kappa <= kappaMax) {
            Vector3D zeros;
            return zeros;
            
        }else{
            double absDxi1Inv = 1 / absDxi;
            double PDphi_PcosDphi = -1 / std::sqrt(1 - std::pow(std::cos(Dphi), 2));
            double u = -absDxi1Inv * PDphi_PcosDphi;
            
            p1 = xi.ort(-xip1) / (absDxi * absDxip1);
            p2 = -xip1.ort(xi) / (absDxi * absDxip1);

            double s = Dphi / (absDxi * absDxi);
            Vector3D ones(1, 1);
            Vector3D ki = u * (-p1 - p2) - (s * ones);
            Vector3D kim1 = u * p2 - (s * ones);
            Vector3D kip1 = u * p1;

            gradient = 0.25 * kim1 + 0.5 * ki + 0.25 * kip1;
            // gradient = ki;

            // -------- Just For Debug
            // std::cout << "theta:" << Dphi << " dist:" << absDxi << " kappa:" << kappa << " grad:" << gradient << std::endl;
            // ------------------------

            if (std::isnan(gradient.getX()) || 
                std::isnan(gradient.getY()) || 
                std::isnan(gradient.getZ())
            ) {
                std::cout << "nan values in curvature term" << std::endl;
                Vector3D zeros;
                return zeros;
            }

            gradient.clamp(-gradMax, gradMax);

            // -------- Just For Debug
            // std::cout << "theta:" << Dphi << " dist:" << absDxi << " kappa:" << kappa << " grad:" << gradient << std::endl;
            // ------------------------

            return gradient;
        }
    }else {
        std::cout << "abs values not larger than 0" << std::endl;
        Vector3D zeros;
        return zeros;
    }

}

void Smoother::updateGradient(){
    AgentSmoothInfo* agent;

    double x, y, z, bound;
    std::vector<KDTreeRes*> resList;
    KDTreeRes* res;    
    Vector3D obs;

    for (auto iter : agentMap)
    {
        agent = iter.second;

        // The First And Last Point Are Fixed
        for (size_t i = 0; i < agent->pathXYZ.size(); i++)
        {
            Vector3D grad;

            if (i <= 1 || i >= agent->pathXYZ.size()-2)
            {
                agent->grads[i] = grad;
                continue;
            }
            
            Vector3D xim2 = agent->pathXYZ[i-2];
            Vector3D xim1 = agent->pathXYZ[i-1];
            Vector3D xi = agent->pathXYZ[i];
            Vector3D xip1 = agent->pathXYZ[i+1];
            Vector3D xip2 = agent->pathXYZ[i+2];

            if (wSmoothness != 0.0)
            {
                Vector3D smooth_grad = getSmoothessGradent(xim2, xim1, xi, xip1, xip2);
                // std::cout << "idx:" << i << " Smooth grad:" << smooth_grad << std::endl;

                grad = grad - wSmoothness * smooth_grad;
            }
            
            if (wCurvature != 0.0)
            {
                Vector3D curvature_grad = getCurvatureGradent(xim1, xi, xip1);
                // std::cout << "idx:" << i << " Curvature grad:" << curvature_grad << std::endl;

                grad = grad - wCurvature * curvature_grad;
            }
            
            if (wObstacle != 0.0)
            {
                x = xi.getX();
                y = xi.getY();
                z = xi.getZ();
                resList.clear();

                Vector3D obs_grad;
                double conflict_num = 0.0;

                for (auto iter2 : agentMap)
                {
                    if (agent->agentIdx == iter2.second->agentIdx)
                    {
                        continue;
                    }

                    bound = agent->radius + iter2.second->radius;
                    iter2.second->pathTree->nearest_range(x, y, z, bound, resList);

                    for (size_t j = 0; j < resList.size(); j++)
                    {
                        res = resList[j];
                        obs = Vector3D(res->x, res->y, res->z);
                        Vector3D point_grad = getObscaleGradent(xi, obs, bound);

                        obs_grad = obs_grad + point_grad;
                        conflict_num += 1.0;

                        delete res;
                    }

                }

                obs_grad = obs_grad / conflict_num;
                grad = grad - wObstacle * obs_grad;
            }

            agent->grads[i] = grad;
        }
    }
}

void Smoother::smoothPath(size_t updateTimes){
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

                agent->pathXYZ[i] = xyz + alpha * grad;
            }
        }

        step += 1;
    }
    
}

DetailPath Smoother::paddingPath(
        DetailPath& detailPath, 
        std::tuple<double, double, double> startPadding,
        std::tuple<double, double, double> endPadding
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
            newPath.emplace_back(std::make_tuple(x, y, z, length));
        }

        newPath.emplace_back(std::make_tuple(
            x + xStart_shift, 
            y + yStart_shift, 
            z + zStart_shift, 
            length + start_shift
        ));

        if (i == detailPath.size()-1)
        {
            newPath.emplace_back(std::make_tuple(
                x + xStart_shift + xEnd_shift, 
                y + yStart_shift + yEnd_shift, 
                z + zStart_shift + zEnd_shift, 
                length + start_shift + end_shift
            ));
        }
    }

    return newPath;
}