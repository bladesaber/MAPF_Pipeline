#include "groupObjSolver.h"

namespace PlannerNameSpace{

Path_XYZRL sampleDetailPath(Path_XYZR& path_xyzr, double stepLength){
    Path_XYZRL new_detail_path;

    double lastX, lastY, lastZ, lastRadius;
    double curX, curY, curZ, curRadius;

    std::tie(lastX, lastY, lastZ, lastRadius) = path_xyzr[0];

    if (path_xyzr.size() == 1){
        new_detail_path.emplace_back(std::make_tuple(
            lastX, lastY, lastZ, lastRadius, 0
        ));
        return new_detail_path;
    }

    double distance, real_stepLength;
    size_t num;
    double vecX, vecY, vecZ, vecRaiuds;
    double cur_length = 0.0;

    for (size_t i = 1; i < path_xyzr.size(); i++){
        std::tie(curX, curY, curZ, curRadius) = path_xyzr[i];

        distance = norm2_distance(lastX, lastY, lastZ, curX, curY, curZ);
        if (distance < stepLength){
            continue;
        }
        
        num = std::ceil(distance / stepLength);
        real_stepLength = distance / (double)num;
        vecX = (curX - lastX) / distance;
        vecY = (curY - lastY) / distance;
        vecZ = (curZ - lastZ) / distance;
        vecRaiuds = (curRadius - lastRadius) / (double)num;

        // for last point
        if (i == path_xyzr.size() - 1){
            num += 1;
        }

        for (size_t j = 0; j < num; j++)
        {
            new_detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength),
                curRadius + vecRaiuds * j,
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

std::vector<std::pair<size_t, size_t>> MultiObjs_GroupSolver::getSequence_miniumSpanningTree(Instance& instance, std::vector<size_t> locs){
    // TODO 这里只使用了启发式作为度量，其实并不完备
    // 这里也不是标准意义上的最小生成树，主要是因为我考虑了实际路径生成中有新的信息，所以没使用最小生成树

    int obj_num = locs.size();
    Eigen::MatrixXd heruristicCost_m(obj_num, obj_num);
    heruristicCost_m = heruristicCost_m.setOnes() * DBL_MAX;

    for (size_t i = 0; i < obj_num; i++){
        for (size_t j = i + 1; j < obj_num; j++){
            double cost = instance.getManhattanDistance(locs[i], locs[j]);
            heruristicCost_m(i, j) = cost;
            heruristicCost_m(j, i) = cost;
        }
    }

    // std::cout << heruristicCost_m << std::endl;

    Eigen::VectorXi isExplored(obj_num);
    isExplored.setZero();

    Eigen::MatrixXd::Index minRow, minCol;
    heruristicCost_m.minCoeff(&minRow, &minCol);
    isExplored[minRow] = 1;
    isExplored[minCol] = 1;

    std::vector<std::pair<size_t, size_t>> treeLinks;
    treeLinks.emplace_back(std::make_pair(minRow, minCol));

    for (size_t i = 0; i < obj_num-2; i++){
        
        int select_row = -1;
        int select_col = -1;
	    double min_cost = DBL_MAX;

        for (size_t j = 0; j < obj_num; j++){
            if (isExplored[j] == 1){
                continue;
            }
            
            // double row_cost = heruristicCost_m.block(j, 0, j, obj_num).minCoeff(&minRow, &minCol);
            // double row_cost = heruristicCost_m.row(j).minCoeff(&minRow, &minCol);
            for (size_t k=0; k < obj_num; k++){
                if (isExplored[k] == 0){
                    continue;
                }

                double row_cost = heruristicCost_m(j, k);
                // std::cout << " j:" << j << " k:" << k << " row_cost:" << row_cost << std::endl;
                if (row_cost < min_cost){
                    min_cost = row_cost;
                    select_row = j;
                    select_col = k;
                }
            }
        }

        isExplored[select_row] = 1;
        treeLinks.emplace_back(std::make_pair(select_row, select_col));
    }
    
    return treeLinks;
}

}