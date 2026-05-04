//===--- test_omp_target_offload_env_MANDATORY.c --------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks for offloading behavior when OMP_TARGET_OFFLOAD is set to MANDATORY.
//
// See test_omp_target_offload_env_DEFAULT.c for details.
//
////===--------------------------------------------------------------------------------------------===//

#define EXPECTED_POLICY MANDATORY

#include "test_omp_target_offload_env_DEFAULT.c"
