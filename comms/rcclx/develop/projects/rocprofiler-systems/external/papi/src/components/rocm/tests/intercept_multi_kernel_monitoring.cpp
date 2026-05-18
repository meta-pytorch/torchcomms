/**
 * @file   intercept_multi_kernel_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#include "multi_kernel_monitoring.h"

int main(int argc, char *argv[])
{
    setenv("ROCP_HSA_INTERCEPT", "1", 1);

    multi_kernel(argc, argv);

    test_pass(__FILE__);
    return 0;
}
