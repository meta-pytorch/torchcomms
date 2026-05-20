/**
 * @file   intercept_single_thread_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#include "single_thread_monitoring.h"

int main(int argc, char *argv[])
{
    setenv("ROCP_HSA_INTERCEPT", "1", 1);

    single_thread(argc, argv);

    return 0;
}
