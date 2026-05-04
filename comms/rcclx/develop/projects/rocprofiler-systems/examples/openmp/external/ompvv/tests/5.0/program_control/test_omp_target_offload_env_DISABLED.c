//===--- test_omp_target_offload_env_DISABLED.c ----------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks for offloading behavior when OMP_TARGET_OFFLOAD is set to DISABLED.
//
// See test_omp_target_offload_env_DEFAULT.c for details.
//
////===--------------------------------------------------------------------------------------------===//

#define EXPECTED_POLICY DISABLED

#include "./test_omp_target_offload_env_DEFAULT.c"
