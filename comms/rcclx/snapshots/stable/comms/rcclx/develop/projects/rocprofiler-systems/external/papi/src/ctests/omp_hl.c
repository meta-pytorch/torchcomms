#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "papi.h"
#include "papi_test.h"
#include "do_loops.h"

int main( int argc, char **argv )
{
   int retval, i;
   int quiet = 0;
   char* region_name;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   region_name = "do_flops";

   #pragma omp parallel
   #pragma omp for
   for ( i = 1; i <= 4; ++i ) {
      int tid;
      tid = omp_get_thread_num();

      if ( !quiet ) {
         printf("\nThread %d: instrument flops\n", tid);
      }

      retval = PAPI_hl_region_begin(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_hl_region_end(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
      }

   }

   region_name = "do_flops_2";
   #pragma omp parallel
   #pragma omp for
   for ( i = 1; i <= 4; ++i ) {
      int tid;
      tid = omp_get_thread_num();

      if ( !quiet ) {
         printf("\nThread %d: instrument flops_2\n", tid);
      }

      retval = PAPI_hl_region_begin(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_hl_region_end(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
      }
   }

   test_hl_pass( __FILE__ );

   return 0;
}