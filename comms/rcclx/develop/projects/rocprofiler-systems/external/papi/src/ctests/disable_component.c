/*
 * File:    disable_component.c
 * Author:  Vince Weaver
 *              vweaver1@eecs.utk.edu
 */

/*
  This tests the functionality of PAPI_disable_component()
*/

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"


int
main( int argc, char **argv )
{
   int retval;
   const PAPI_component_info_t* cmpinfo;
   int numcmp, cid, active_components=0;

   /* Set TESTS_QUIET variable */
   tests_quiet( argc, argv );

   /* Disable All Compiled-in Components */
   numcmp = PAPI_num_components(  );

   if (!TESTS_QUIET) printf("Compiled-in components:\n");
   for( cid = 0; cid < numcmp; cid++ ) {
      cmpinfo = PAPI_get_component_info( cid );

      if (!TESTS_QUIET) {
         printf( "Name:   %-23s %s\n", cmpinfo->name, cmpinfo->description);
      }

      retval=PAPI_disable_component( cid );
      if (retval!=PAPI_OK) {
	 test_fail(__FILE__,__LINE__,"Error disabling component",retval);
      }
   }


   /* Initialize the library */
   retval = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retval != PAPI_VER_CURRENT ) {
      test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
   }

   /* Try to disable after init, should fail */
   retval=PAPI_disable_component( 0 );
   if (retval==PAPI_OK) {
      test_fail( __FILE__, __LINE__, "PAPI_disable_component should fail", 
		 retval );
   }

   if (!TESTS_QUIET) printf("\nAfter init components:\n");
   for( cid = 0; cid < numcmp; cid++ ) {

      cmpinfo = PAPI_get_component_info( cid );

      if (!TESTS_QUIET) {
	printf( "%d %d Name:   %-23s %s\n",
		cid,
		PAPI_get_component_index((char *)cmpinfo->name),
		cmpinfo->name ,cmpinfo->description);

      }

      if (cid!=PAPI_get_component_index((char *)cmpinfo->name)) {
         test_fail( __FILE__, __LINE__, "PAPI_get_component_index mismatch",
		 2 );
      }


      if (cmpinfo->disabled) {
	 if (!TESTS_QUIET) {
            printf("   \\-> Disabled: %s\n",cmpinfo->disabled_reason);
	 }
      } else {
	 active_components++;
      }
   }

   if (active_components>0) {
      test_fail( __FILE__, __LINE__, "too many active components", retval );
   }

	test_pass( __FILE__ );

	return PAPI_OK;
}
