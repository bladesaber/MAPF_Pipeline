#ifndef MAPF_PIPELINE_AUX_DUBINS_H
#define MAPF_PIPELINE_AUX_DUBINS_H

#include <math.h>
#include <tuple>
#include <map>

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

int dubins_intermediate_results(
    DubinsIntermediateResults* in, double q0[3], double q1[3], double rho
)
{
    /*
    q0: x, y, theta
    q1: x, y, theta
    */

    double dx, dy, D, d, theta, alpha, beta;
    if( rho <= 0.0 ) {
        return EDUBBADRHO;
    }

    dx = q1[0] - q0[0];
    dy = q1[1] - q0[1];
    D = sqrt( dx * dx + dy * dy );
    d = D / rho;
    theta = 0;

    /* test required to prevent domain errors if dx=0 and dy=0 */
    if(d > 0) {
        theta = mod2pi(atan2( dy, dx ));
    }
    alpha = mod2pi(q0[2] - theta);
    beta  = mod2pi(q1[2] - theta);

    in->alpha = alpha;
    in->beta  = beta;
    in->d     = d;
    in->sa    = sin(alpha);
    in->sb    = sin(beta);
    in->ca    = cos(alpha);
    in->cb    = cos(beta);
    in->c_ab  = cos(alpha - beta);
    in->d_sq  = d * d;

    return EDUBOK;
}

void compute_dubins_pathLength(DubinsPath* path)
{
    double length1, length2, length3;
    length1 = std::get<0>(path->param) * path->rho;
    length2 = std::get<1>(path->param) * path->rho;
    length3 = std::get<2>(path->param) * path->rho;

    path->total_length += length1 + length2 + length3;
    path->lengths = std::make_tuple(length1, length2, length3);
}

int dubins_LSL(DubinsIntermediateResults* in, double out[3]) 
{
    double tmp0, tmp1, p_sq;
    
    tmp0 = in->d + in->sa - in->sb;
    p_sq = 2 + in->d_sq - (2*in->c_ab) + (2 * in->d * (in->sa - in->sb));

    if(p_sq >= 0) {
        tmp1 = atan2( (in->cb - in->ca), tmp0 );
        out[0] = mod2pi(tmp1 - in->alpha);
        out[1] = sqrt(p_sq);
        out[2] = mod2pi(in->beta - tmp1);
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_RSL(DubinsIntermediateResults* in, double out[3]) 
{
    double p_sq = -2 + in->d_sq + (2 * in->c_ab) - (2 * in->d * (in->sa + in->sb));
    if( p_sq >= 0 ) {
        double p    = sqrt(p_sq);
        double tmp0 = atan2( (in->ca + in->cb), (in->d - in->sa - in->sb) ) - atan2(2.0, p);
        out[0] = mod2pi(in->alpha - tmp0);
        out[1] = p;
        out[2] = mod2pi(in->beta - tmp0);
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_RLR(DubinsIntermediateResults* in, double out[3]) 
{
    double tmp0 = (6. - in->d_sq + 2*in->c_ab + 2*in->d*(in->sa - in->sb)) / 8.;
    double phi  = atan2( in->ca - in->cb, in->d - in->sa + in->sb );
    if( fabs(tmp0) <= 1) {
        double p = mod2pi((2*M_PI) - acos(tmp0) );
        double t = mod2pi(in->alpha - phi + mod2pi(p/2.));
        out[0] = t;
        out[1] = p;
        out[2] = mod2pi(in->alpha - in->beta - t + mod2pi(p));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_LRL(DubinsIntermediateResults* in, double out[3])
{
    double tmp0 = (6. - in->d_sq + 2*in->c_ab + 2*in->d*(in->sb - in->sa)) / 8.;
    double phi = atan2( in->ca - in->cb, in->d + in->sa - in->sb );
    if( fabs(tmp0) <= 1) {
        double p = mod2pi( 2*M_PI - acos( tmp0) );
        double t = mod2pi(-in->alpha - phi + p/2.);
        out[0] = t;
        out[1] = p;
        out[2] = mod2pi(mod2pi(in->beta) - in->alpha -t + mod2pi(p));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_RSR(DubinsIntermediateResults* in, double out[3]) 
{
    double tmp0 = in->d - in->sa + in->sb;
    double p_sq = 2 + in->d_sq - (2 * in->c_ab) + (2 * in->d * (in->sb - in->sa));
    if( p_sq >= 0 ) {
        double tmp1 = atan2( (in->ca - in->cb), tmp0 );
        out[0] = mod2pi(in->alpha - tmp1);
        out[1] = sqrt(p_sq);
        out[2] = mod2pi(tmp1 -in->beta);
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_LSR(DubinsIntermediateResults* in, double out[3]) 
{
    double p_sq = -2 + (in->d_sq) + (2 * in->c_ab) + (2 * in->d * (in->sa + in->sb));
    if( p_sq >= 0 ) {
        double p    = sqrt(p_sq);
        double tmp0 = atan2( (-in->ca - in->cb), (in->d + in->sa + in->sb) ) - atan2(-2.0, p);
        out[0] = mod2pi(tmp0 - in->alpha);
        out[1] = p;
        out[2] = mod2pi(tmp0 - mod2pi(in->beta));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

void compute_CircleInfo(
    double x, double y, double rho, double radian, double move_radian,
    double center[2], double range[2],
    SegmentType circleType, bool last_circle
){
    if (circleType == L_SEG)
    {
        center[0] = x - sin(radian) * rho;
        center[1] = y + cos(radian) * rho;

        if (last_circle)
        {
            range[1] = radian - M_PI / 2.0;
            range[0] = range[1] - move_radian;

        }else{
            range[0] = radian - M_PI / 2.0;
            range[1] = range[0] + move_radian;
        }

    }else if (circleType == R_SEG)
    {
        center[0] = x + sin(radian) * rho;
        center[1] = y - cos(radian) * rho;

        if (last_circle){
            range[1] = radian + M_PI / 2.0;
            range[0] = range[1] + move_radian;

        }else{
            range[0] = radian + M_PI / 2.0;
            range[1] = range[0] - move_radian;
        }
    }
}

void compute_dubins_info(DubinsPath* path)
{
    double center[2];
    double ranges[2];

    compute_CircleInfo(
        std::get<0>(path->q0),
        std::get<1>(path->q0),
        path->rho,
        std::get<2>(path->q0),
        std::get<0>(path->param),
        center,
        ranges,
        path->segmentTypes[0],
        false
    );
    path->start_center = std::make_tuple(center[0], center[1]);
    path->start_range = std::make_tuple(ranges[0], ranges[1]);
    path->line_sxy = std::make_tuple(
        center[0] + path->rho * cos(ranges[1]), 
        center[1] + path->rho * sin(ranges[1])
    );

    compute_CircleInfo(
        std::get<0>(path->q1),
        std::get<1>(path->q1),
        path->rho,
        std::get<2>(path->q1),
        std::get<2>(path->param),
        center,
        ranges,
        path->segmentTypes[2],
        true
    );
    path->final_center = std::make_tuple(center[0], center[1]);
    path->final_range = std::make_tuple(ranges[0], ranges[1]);
    path->line_fxy = std::make_tuple(
        center[0] + path->rho * cos(ranges[0]), 
        center[1] + path->rho * sin(ranges[0])
    );
}

int compute_dubins_path(
    DubinsPath* result, 
    std::tuple<double, double, double> xyz_s, 
    std::tuple<double, double, double> xyz_t, 
    double rho, DubinsPathType pathType
)
{
    int errcode;
    DubinsIntermediateResults in;
    double q0[3] = {
        std::get<0>(xyz_s), std::get<1>(xyz_s), std::get<2>(xyz_s),
    };
    double q1[3] = {
        std::get<0>(xyz_t), std::get<1>(xyz_t), std::get<2>(xyz_t),
    };
    errcode = dubins_intermediate_results(&in, q0, q1, rho);

    if(errcode == EDUBOK) {
        double params[3];
        switch (pathType)
        {
        case LSL:
            errcode = dubins_LSL(&in, params);
            result->segmentTypes[0] = L_SEG;
            result->segmentTypes[1] = S_SEG;
            result->segmentTypes[2] = L_SEG;
            break;

        case RSL:
            errcode = dubins_RSL(&in, params);
            result->segmentTypes[0] = R_SEG;
            result->segmentTypes[1] = S_SEG;
            result->segmentTypes[2] = L_SEG;
            break;

        case LSR:
            errcode = dubins_LSR(&in, params);
            result->segmentTypes[0] = L_SEG;
            result->segmentTypes[1] = S_SEG;
            result->segmentTypes[2] = R_SEG;
            break;

        case RSR:
            errcode = dubins_RSR(&in, params);
            result->segmentTypes[0] = R_SEG;
            result->segmentTypes[1] = S_SEG;
            result->segmentTypes[2] = R_SEG;
            break;

        // case LRL:
        //     errcode = dubins_LRL(&in, params);
        //     result->segmentTypes[0] = L_SEG;
        //     result->segmentTypes[1] = R_SEG;
        //     result->segmentTypes[2] = L_SEG;
        //     break;

        // case RLR:
        //     errcode = dubins_RLR(&in, params);
        //     result->segmentTypes[0] = R_SEG;
        //     result->segmentTypes[1] = L_SEG;
        //     result->segmentTypes[2] = R_SEG;
        //     break;
        
        default:
            errcode = EDUBNOPATH;
            break;
        }

        if(errcode == EDUBOK){
            result->q0 = std::make_tuple(q0[0], q0[1], q0[2]);
            result->q1 = std::make_tuple(q1[0], q1[1], q1[2]);
            result->param = std::make_tuple(params[0], params[1], params[2]);
            result->rho = rho;
            result->type = pathType;

            compute_dubins_pathLength(result);
            // compute_dubins_info(result);
        }
    }
    return errcode;
}

void sample_CircleSegment(
    double length, double rho, 
    double start_radian, double center_x, double center_y, 
    SegmentType circleType, double xy[2]
){
    double radian = length / rho;
    double end_radian;

    if (circleType == L_SEG)
    {
        end_radian = start_radian + radian;
        xy[0] = center_x + rho * cos(end_radian);
        xy[1] = center_y + rho * sin(end_radian);

    }else if(circleType == R_SEG){
        end_radian = start_radian - radian;
        xy[0] = center_x + rho * cos(end_radian);
        xy[1] = center_y + rho * sin(end_radian);
    }
}

void sample_LineSegment(
    double length, double radian, double sx, double sy, double xy[2]
){
    xy[0] = sx + length * cos(radian);
    xy[1] = sy + length * sin(radian);
}

std::list<std::pair<double, double>> sample_dubins_path(DubinsPath* path, size_t sample_size){
    double step_length = path->total_length / (double)sample_size;
    double line_x = std::get<0>(path->line_sxy);
    double line_y = std::get<1>(path->line_sxy);

    double line_radian = atan2(
        std::get<1>(path->line_fxy) - std::get<1>(path->line_sxy),
        std::get<0>(path->line_fxy) - std::get<0>(path->line_sxy)
    );

    double stage1 = std::get<0>(path->lengths);
    double stage2 = stage1 + std::get<1>(path->lengths);

    double cur_length = 0.0;
    double xy[2];
    std::list<std::pair<double, double>> sample_waypoints;

    for (size_t i = 0; i < sample_size + 1; i++)
    {
        cur_length = i * step_length;
        if (cur_length < stage1)
        {
            sample_CircleSegment(
                cur_length, 
                path->rho, std::get<0>(path->start_range),
                std::get<0>(path->start_center), std::get<1>(path->start_center),
                path->segmentTypes[0], xy
            );

        }
        else if (cur_length < stage2)
        {
            sample_LineSegment(
                cur_length - stage1, 
                line_radian, line_x, line_y, xy
            );

        }
        else{
            sample_CircleSegment(
                cur_length - stage2, 
                path->rho, std::get<0>(path->final_range),
                std::get<0>(path->final_center), std::get<1>(path->final_center),
                path->segmentTypes[2], xy
            );
        }
        
        sample_waypoints.push_back(std::make_pair(xy[0], xy[1]));

    }
    
    return sample_waypoints;
}

#endif /* MAPF_PIPELINE_AUX_DUBINS_H */