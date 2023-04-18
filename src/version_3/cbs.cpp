#include "cbs.h"

void CBS::pushNode(CBSNode* node){
    open_list.push(node);
}

CBSNode* CBS::popNode(){
    CBSNode* node = open_list.top();
    open_list.pop();
    return node;
}

DetailPath CBS::sampleDetailPath(Path& path, Instance& instance, double stepLength){
    DetailPath detail_path;

    if (path.size() == 0)
    {
        return detail_path;
    }
    
    double lastX, lastY, lastZ;
    std::tie(lastX, lastY, lastZ) = instance.getCoordinate(path[0]);

    double curX, curY, curZ;
    double vecX, vecY, vecZ;
    double distance, real_stepLength;
    size_t num;
    for (size_t i = 1; i < path.size(); i++)
    {
        std::tie(curX, curY, curZ) = instance.getCoordinate(path[i]);

        distance = norm2_distance(
            lastX, lastY, lastZ,
            curX, curY, curZ
        );

        num = std::ceil(distance / stepLength);
        real_stepLength = distance / (double)num;

        vecX = (curX - lastX) / distance;
        vecY = (curY - lastY) / distance;
        vecZ = (curZ - lastZ) / distance;

        // for last point
        if (i == path.size() - 1)
        {
            num += 1;
        }
        
        // std::cout << "lastX: " << lastX << " lastY: " << lastY << " lastZ: " << lastZ << std::endl;
        // std::cout << "curX: " << curX << " curY: " << curY << " curZ: " << curZ << std::endl;
        // std::cout << "vecX: " << vecX << " vecY: " << vecY << " vecZ: " << vecZ << std::endl;

        for (size_t j = 0; j < num; j++)
        {
            // std::cout << "[" <<  lastX + vecX * (j * real_stepLength) << ", " 
            //           << lastY + vecY * (j * real_stepLength) << ", "
            //           << lastZ + vecZ * (j * real_stepLength) << "]" << std::endl;

            detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength)
            ));
        }

        lastX = curX;
        lastY = curY;
        lastZ = curZ;
    }

    return detail_path;
}

void CBS::findConflictFromTree(KDTreeWrapper& tree, DetailPath& path, double bound){
    double x, y, z;
    double treeX, treeY, treeZ;

    for (size_t i = 0; i < path.size(); i++)
    {
        std::tie(x, y, z) = path[i];
        std::tie(treeX, treeY, treeZ) = tree.nearest(x, y, z);

        

    }
}
