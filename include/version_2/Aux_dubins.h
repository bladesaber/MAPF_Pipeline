#ifndef MAPF_PIPELINE_AUX_DUBINS_H
#define MAPF_PIPELINE_AUX_DUBINS_H

#include "Aux_common.h"
#include "Aux_utils.h"

enum DubinsErrorCodes {
    EDUBOK,         /* No error */
    EDUBCOCONFIGS,  /* Colocated configurations */
    EDUBPARAM,      /* Path parameterisitation error */
    EDUBBADRHO,     /* the rho value is invalid */
    EDUBNOPATH      /* no connection between configurations with this word */
};

enum DubinsPathType {
    LSL, LSR, RSL, RSR, RLR, LRL,
};

enum SegmentType
{
    L_SEG = 0,
    S_SEG = 1,
    R_SEG = 2
};

typedef struct 
{
    /* the initial configuration (x, y, theta) */
    std::tuple<double, double, double> q0;
    /* the final configuration (x, y, theta) */
    std::tuple<double, double, double> q1;
    /* the radius of wheel */
    double rho;

    // for L or R: radian
    // for S: length / radian
    std::tuple<double, double, double> param;
    /* the path type described */    
    DubinsPathType type;

    std::tuple<double, double, double> lengths;
    double total_length = 0.0;

    SegmentType segmentTypes[3];
    std::tuple<double, double> start_center;
    std::tuple<double, double> start_range;

    std::tuple<double, double> final_center;
    std::tuple<double, double> final_range;

    std::tuple<double, double> line_sxy;
    std::tuple<double, double> line_fxy;

} DubinsPath;

/* The segment types for each of the Path types */
// static std::map<DubinsPathType, std::tuple<SegmentType, SegmentType, SegmentType>> DubinsMap{
//     {LSL, { L_SEG, S_SEG, L_SEG }},
//     {LSR, { L_SEG, S_SEG, R_SEG }},
//     {RSL, { R_SEG, S_SEG, L_SEG }},
//     {RSR, { R_SEG, S_SEG, R_SEG }},
//     {RLR, { R_SEG, L_SEG, R_SEG }},
//     {LRL, { L_SEG, R_SEG, L_SEG }},
// };

typedef struct 
{
    double alpha;
    double beta;
    double d;
    double sa;
    double sb;
    double ca;
    double cb;
    double c_ab;
    double d_sq;
} DubinsIntermediateResults;

int dubins_intermediate_results(DubinsIntermediateResults* in, double q0[3], double q1[3], double rho);

void compute_dubins_pathLength(DubinsPath* path);

int dubins_LSL(DubinsIntermediateResults* in, double out[3]);

int dubins_RSL(DubinsIntermediateResults* in, double out[3]);

int dubins_RLR(DubinsIntermediateResults* in, double out[3]);

int dubins_LRL(DubinsIntermediateResults* in, double out[3]);

int dubins_RSR(DubinsIntermediateResults* in, double out[3]);

int dubins_LSR(DubinsIntermediateResults* in, double out[3]);

void compute_CircleInfo(
    double x, double y, double rho, double radian, double move_radian,
    double center[2], double range[2],
    SegmentType circleType, bool last_circle
);

void compute_dubins_info(DubinsPath* path);

int compute_dubins_path(
    DubinsPath* result, 
    std::tuple<double, double, double> xyz_s, 
    std::tuple<double, double, double> xyz_t, 
    double rho, DubinsPathType pathType
);

void sample_CircleSegment(
    double length, double rho, 
    double start_radian, double center_x, double center_y, 
    SegmentType circleType, double xy[2]
);

void sample_LineSegment(
    double length, double radian, double sx, double sy, double xy[2]
);

std::list<std::pair<double, double>> sample_dubins_path(DubinsPath* path, size_t sample_size);

#endif /* MAPF_PIPELINE_AUX_DUBINS_H */