#ifndef SIMP2_LIB_H
#define SIMP2_LIB_H

#include "sde_lib.h"
#include "sde_lib.hpp"

class Simple{
public:
    Simple();
    double simple_compute(double x);
    static constexpr const char *ev_names[4] = {"COMPUTED_VALUE", "TOTAL_ITERATIONS", "LOW_WATERMARK_REACHED", "HIGH_WATERMARK_REACHED" };

private:
    static long long int counter_accessor_function( double *param );
    double comp_value;
    long long int total_iter_cnt, low_wtrmrk, high_wtrmrk;
};

#endif
