#include "utils.h"

double fmodr( double x, double y)
{
    return x - y * floor(x / y);
}

double mod2pi(double theta)
{
    return fmodr( theta, 2 * M_PI );
}

double mod2singlePi(double theta){
    return fmodr(theta + M_PI, 2 * M_PI ) - M_PI;
}

double rad2degree(double theta){
    return theta / M_PI * 180.0;
}

std::tuple<double, double, double> polar3D_to_vec3D(double alpha, double beta, double length){
    double dz = length * sin(beta);
    double dl = length * cos(beta);
    double dx = dl * cos(alpha);
    double dy = dl * sin(alpha);
    return std::make_tuple(dx, dy, dz);
}

std::tuple<double, double> vec3D_to_polar3D(double vec_x, double vec_y, double vec_z){
    double alpha = mod2pi(atan2(vec_y, vec_x));
    double length = sqrt( pow(vec_x, 2) + pow(vec_y, 2));
    double beta = atan2(vec_z, length);
    return std::make_tuple(alpha, beta);
}

double point2LineDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
){
    // vector_a * vector_b
    double vecSum = (lineEnd_x - lineStart_x) * (point_x - lineStart_x) + 
                    (lineEnd_y - lineStart_y) * (point_y - lineStart_y) + 
                    (lineEnd_z - lineStart_z) * (point_z - lineStart_z);

    // length of line
    double length1 = sqrt(
        pow(lineEnd_x - lineStart_x, 2) + 
        pow(lineEnd_y - lineStart_y, 2) + 
        pow(lineEnd_z - lineStart_z, 2)
    );

    double length2 = sqrt(
        pow(point_x - lineStart_x, 2) + 
        pow(point_y - lineStart_y, 2) + 
        pow(point_z - lineStart_z, 2)
    );

    double cosTheta = vecSum / (length1 * length2);
    double sinTheta = sqrt(1.0 - pow(cosTheta, 2));

    double dist = length2 * sinTheta;

    return dist;
}

double norm2_distance(
    double x0, double y0, double z0,
    double x1, double y1, double z1
){
    return sqrt(
        pow(x0 - x1, 2) + 
        pow(y0 - y1, 2) + 
        pow(z0 - z1, 2)
    );
}

double point2LineSegmentDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
){
    double norm0 = norm2_distance(
        lineStart_x, lineStart_y, lineStart_z,
        point_x, point_y, point_z
    );
    double norm1 = norm2_distance(
        lineEnd_x, lineEnd_y, lineEnd_z,
        point_x, point_y, point_z
    );

    double startX, startY, startZ;
    double endX, endY, endZ;
    double length2;
    if (norm0 > norm1)
    {
        startX = lineStart_x;
        startY = lineStart_y;
        startZ = lineStart_z;
        endX = lineEnd_x;
        endY = lineEnd_y;
        endZ = lineEnd_z;
        length2 = norm0;

    }else{
        startX = lineEnd_x;
        startY = lineEnd_y;
        startZ = lineEnd_z;
        endX = lineStart_x;
        endY = lineStart_y;
        endZ = lineStart_z;
        length2 = norm1;
    }

    // vector_a * vector_b
    double vecSum = (endX - startX) * (point_x - startX) + 
                    (endY - startY) * (point_y - startY) + 
                    (endZ - startZ) * (point_z - startZ);

    double line_length = norm2_distance(
        endX, endY, endZ,
        startX, startY, startZ
    );

    double cosLength = abs(vecSum / line_length);

    // std::cout << "Line Length: " << line_length << std::endl;
    // std::cout << "start->point Length: " << length2 << std::endl;
    // std::cout << "cosin Length: " << cosLength << std::endl;

    double dist;
    if (cosLength > line_length)
    {
        dist = std::min(norm0, norm1);

    }else{
        dist = sqrt(pow(length2, 2) - pow(cosLength, 2));
    }

    return dist;
}
