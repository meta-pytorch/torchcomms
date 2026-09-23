#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include "papi.h"
#include "papi_test.h"
#include "do_loops.h"

#define NUM_THREADS 4

typedef struct papi_args
{
   long tid;
   int quiet;
} papi_args_t;

void *CallMatMul(void *args)
{
   long tid;
   int retval, quiet;
   char* region_name;

   papi_args_t* papi_args = (papi_args_t*)args;
   tid = (*papi_args).tid;
   quiet = (*papi_args).quiet;
   region_name = "do_flops";

   if ( !quiet ) {
      printf("\nThread %ld: instrument flops\n", tid);
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

   pthread_exit(NULL);
}

int main( int argc, char **argv )
{
   pthread_t threads[NUM_THREADS];
   papi_args_t args[NUM_THREADS];
   int rc;
   long t;
   int quiet = 0;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   for( t = 0; t < NUM_THREADS; t++) {
      args[t].tid = t;
      args[t].quiet = quiet;
      rc = pthread_create(&threads[t], NULL, CallMatMul, (void *)&args[t]);
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

   for( t = 0; t < NUM_THREADS; t++) {
      pthread_join(threads[t], NULL);
   }


   for( t = 0; t < NUM_THREADS; t++) {
      args[t].tid = t;
      args[t].quiet = quiet;
      rc = pthread_create(&threads[t], NULL, CallMatMul, (void *)&args[t]);
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

   for( t = 0; t < NUM_THREADS; t++) {
      pthread_join(threads[t], NULL);
   }

   test_hl_pass( __FILE__ );

   return 0;
}