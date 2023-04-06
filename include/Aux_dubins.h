#ifndef MAPF_PIPELINE_AUX_DUBINS_H
#define MAPF_PIPELINE_AUX_DUBINS_H

#include <math.h>
#include <tuple>

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

typedef struct 
{
    /* the initial configuration (x, y, theta) */
    std::tuple<double, double, double> q0;
    /* the final configuration (x, y, theta) */
    std::tuple<double, double, double> q1;
    /* the radius of wheel */
    double rho;

    /* radian or length */
    std::tuple<double, double, double> param;
    /* the path type described */    
    DubinsPathType type;

} DubinsPath;

// typedef enum 
// {
//     L_SEG = 0,
//     S_SEG = 1,
//     R_SEG = 2
// } SegmentType;

// /* The segment types for each of the Path types */
// const SegmentType DIRDATA[][3] = {
//     { L_SEG, S_SEG, L_SEG },
//     { L_SEG, S_SEG, R_SEG },
//     { R_SEG, S_SEG, L_SEG },
//     { R_SEG, S_SEG, R_SEG },
//     { R_SEG, L_SEG, R_SEG },
//     { L_SEG, R_SEG, L_SEG }
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
            break;

        case RSL:
            errcode = dubins_RSL(&in, params);
            break;

        case LSR:
            errcode = dubins_LSR(&in, params);
            break;

        case RSR:
            errcode = dubins_RSR(&in, params);
            break;

        case LRL:
            errcode = dubins_LRL(&in, params);
            break;

        case RLR:
            errcode = dubins_RLR(&in, params);
            break;
        
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
        }
    }
    return errcode;
}

#endif /* MAPF_PIPELINE_AUX_DUBINS_H */