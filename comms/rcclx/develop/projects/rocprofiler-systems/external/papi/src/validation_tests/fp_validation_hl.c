/* This test runs a "classic" matrix multiply
 * and then runs it again with the inner loop swapped.
 * the swapped version should have better MFLIPS/MFLOPS/IPC and we test that.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"
#include "testcode.h"

int main( int argc, char **argv )
{
   int retval;
   int quiet = 0;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   // Flips classic
   retval = PAPI_hl_region_begin("matrix_multiply_classic");
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
   }
   if ( !quiet ) {
      printf("flops_float_matrix_matrix_multiply()\n");
   }
   flops_float_matrix_matrix_multiply();
   retval = PAPI_hl_region_end("matrix_multiply_classic");
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
   }

   // Flips swapped
   retval = PAPI_hl_region_begin("matrix_multiply_swapped");
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
   }
   if ( !quiet ) {
      printf("flops_float_swapped_matrix_matrix_multiply()\n");
   }
   flops_float_swapped_matrix_matrix_multiply();
   retval = PAPI_hl_region_end("matrix_multiply_swapped");
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
   }

   test_hl_pass( __FILE__ );

   return 0;
}
