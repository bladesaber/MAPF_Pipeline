#include "spanningTree_groupSolver.h"

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

std::vector<std::pair<size_t, size_t>> SpanningTree_GroupSolver::getSequence_miniumSpanningTree(Instance& instance, std::vector<size_t> locs){
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

    std::map<size_t, TreeLeaf*> branchMap;
    for (size_t i=0; i<locs.size(); i++ ){
        branchMap[i] = new TreeLeaf(i);
    }

    Eigen::MatrixXd::Index minRow, minCol;
    int found_valid_num = 0;
        
    std::vector<std::pair<size_t, size_t>> treeLinks;
    while (true) {
        heruristicCost_m.minCoeff(&minRow, &minCol);

        heruristicCost_m(minRow, minCol) = DBL_MAX;
        heruristicCost_m(minCol, minRow) = DBL_MAX;

        // std::cout << "minRow:" << minRow << " minCol:" << minCol << std::endl;

        if ( branchMap[minRow]->isSameSet( branchMap[minCol] ) ) {
            continue;
        }
        
        TreeLeaf* treeLeaf = new TreeLeaf();
        treeLeaf->mergeBranch(branchMap[minRow], branchMap[minCol]);

        delete branchMap[minRow];
        delete branchMap[minCol];
        for (size_t sign: treeLeaf->relative_set){
            branchMap[sign] = treeLeaf;   
        }

        treeLinks.emplace_back(std::make_pair(minRow, minCol));

        found_valid_num += 1;
        if ( found_valid_num >= obj_num - 1 ) {
            delete treeLeaf;
            break;
        }
    }

    std::vector<std::pair<size_t, size_t>> locLinks;
    for (auto iter: treeLinks){
        locLinks.emplace_back(std::make_pair(locs[iter.first], locs[iter.second]));
    }

    return locLinks;
}

void SpanningTree_GroupSolver::copy(std::shared_ptr<SpanningTree_GroupSolver> rhs, bool with_path){
    for (TaskInfo* task : rhs->task_seq){
        TaskInfo* new_obj = new TaskInfo();
        new_obj->link_sign0 = task->link_sign0;
        new_obj->link_sign1 = task->link_sign1;
        new_obj->radius0 = task->radius0;
        new_obj->radius1 = task->radius1;

        if (with_path){
            new_obj->res_path = Path_XYZRL(task->res_path);
        }
            
        this->task_seq.emplace_back(new_obj);
    }
    this->locations = std::vector<size_t>(rhs->locations);
}

void SpanningTree_GroupSolver::updateLocTree(){
    if (setup_tree){
        delete locTree;
    }

    locTree = new KDTree_XYZRL();
    double x, y, z, radius, length;
    for (TaskInfo* task : task_seq)
    {           
        for (size_t i = 0; i < task->res_path.size(); i++){
            std::tie(x, y, z, radius, length) = task->res_path[i];
            locTree->insertNode(0, x, y, z, radius, length);
        }
    }
    setup_tree = true;
}

}