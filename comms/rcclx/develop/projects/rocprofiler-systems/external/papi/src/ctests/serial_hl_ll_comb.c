#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

   /* three iterations with high-level API */
   if ( !quiet ) {
      printf("\nTesting high-level API: do_flops\n");
   }

   for ( i = 1; i < 4; ++i ) {

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

   if ( !quiet ) {
      printf("\nTesting low-level API: do_flops\n");
   }

   long long values[2];
   int EventSet = PAPI_NULL;
   char event_name1[]="appio:::READ_BYTES";
   char event_name2[]="appio:::WRITE_BYTES";

   /* create the eventset */
   retval = PAPI_create_eventset( &EventSet );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
   }

   retval = PAPI_add_named_event( EventSet, event_name1);
   if ( retval != PAPI_OK ) {
      if (!quiet) printf("Couldn't add %s\n",event_name1);
      test_skip(__FILE__,__LINE__,"Couldn't add appio:::READ_BYTES",0);
   }

   retval = PAPI_add_named_event( EventSet, event_name2);
   if ( retval != PAPI_OK ) {
      if (!quiet) printf("Couldn't add %s\n",event_name2);
      test_skip(__FILE__,__LINE__,"Couldn't add appio:::WRITE_BYTES",0);
   }

   /* Start PAPI */
   retval = PAPI_start( EventSet );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   /* Read results */
   retval = PAPI_stop( EventSet, values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
      printf("%s: %lld\n", event_name1, values[0]);
      printf("%s: %lld\n", event_name2, values[1]);
   }

   /* remove results. */
   PAPI_remove_named_event(EventSet,event_name1);
   PAPI_remove_named_event(EventSet,event_name2);

   test_hl_pass( __FILE__ );

   return 0;
}