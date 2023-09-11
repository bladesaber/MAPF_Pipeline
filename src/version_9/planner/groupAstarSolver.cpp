#include "groupAstarSolver.h"

namespace PlannerNameSpace {

    Path_XYZRL sampleDetailPath(Path_XYZR &path_xyzr, double stepLength) {
        Path_XYZRL new_detail_path;

        double lastX, lastY, lastZ, lastRadius;
        double curX, curY, curZ, curRadius;

        std::tie(lastX, lastY, lastZ, lastRadius) = path_xyzr[0];

        if (path_xyzr.size() == 1) {
            new_detail_path.emplace_back(std::make_tuple(
                    lastX, lastY, lastZ, lastRadius, 0
            ));
            return new_detail_path;
        }

        double distance, real_stepLength;
        size_t num;
        double vecX, vecY, vecZ, vecRaiuds;
        double cur_length = 0.0;

        for (size_t i = 1; i < path_xyzr.size(); i++) {
            std::tie(curX, curY, curZ, curRadius) = path_xyzr[i];

            distance = norm2_distance(lastX, lastY, lastZ, curX, curY, curZ);
            if (distance < stepLength && (i < path_xyzr.size() - 1)) {
                continue;
            }

            num = std::ceil(distance / stepLength);
            real_stepLength = distance / (double) num;
            vecX = (curX - lastX) / distance;
            vecY = (curY - lastY) / distance;
            vecZ = (curZ - lastZ) / distance;
            vecRaiuds = (curRadius - lastRadius) / (double) num;

            // for last point
            if (i == path_xyzr.size() - 1) {
                num += 1;
            }

            for (size_t j = 0; j < num; j++) {
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

}