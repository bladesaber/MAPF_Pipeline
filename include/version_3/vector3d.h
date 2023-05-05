#ifndef MAPF_PIPELINE_VECTOR3D_H
#define MAPF_PIPELINE_VECTOR3D_H

#include "common.h"
#include "utils.h"

class Vector3D
{
private:
    /* data */
public:
    inline Vector3D(const double x=0, const double y=0, const double z=0):x(x), y(y), z(z){};
    ~Vector3D(){};

    inline Vector3D operator * (const double k) const {
        return Vector3D(x * k, y * k, z * k);
    }

    inline Vector3D operator / (const float k) const {
        return Vector3D(x / k, y / k, z / k);
    }

    inline Vector3D operator + (const Vector3D& b) const {
        return Vector3D(x + b.x, y + b.y, z + b.z);
    }

    inline Vector3D operator - (const Vector3D& b) const {
        return Vector3D(x - b.x, y - b.y, z - b.z);
    }

    inline Vector3D operator - () const {
        return Vector3D(-x, -y, -z);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector3D& b) {
        os << "(x:" << b.x << " y:" << b.y << " z:" << b.z << ")"; return os; 
    }

    double length() const {
        return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
    }

    double sqlength() const {
        return x*x + y*y + z*z;
    }

    double dot(Vector3D b) {
        return x * b.x + y * b.y + z * b.z;
    }

    inline Vector3D ort(Vector3D b) {
        Vector3D a(this->x, this->y, this->z);
        Vector3D c;
        // multiply b by the dot product of this and b then divide it by b's length
        c = a - b * a.dot(b) / b.sqlength();
        return c;
    }

    void clamp(double lower, double upper){
        x = rangeClamp(x, lower, upper);
        y = rangeClamp(y, lower, upper);
        z = rangeClamp(z, lower, upper);
    }

    inline double getX() {
        return x;
    }
    inline double getY() {
        return y;
    }
    inline double getZ() {
        return z;
    }

private:
    double x;
    double y;
    double z;

};

inline Vector3D operator * (double k, const Vector3D& b) {
  return b * k;
}

#endif