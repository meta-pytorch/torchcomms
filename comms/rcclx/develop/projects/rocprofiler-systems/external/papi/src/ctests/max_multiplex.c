/* this tests attempts to add the maximum number of pre-defined events */
/* to a multiplexed event set.  This tests that we properly set the    */
/* maximum events value.                                               */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int main(int argc, char **argv) {

  int retval,max_multiplex,i,EventSet=PAPI_NULL;
  PAPI_event_info_t info;
  int added=0;
  int events_tried=0;

        /* Set TESTS_QUIET variable */
        tests_quiet( argc, argv );

	/* Initialize the library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_multiplex_init(  );
        if ( retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "Multiplex not supported", 1);
	}

	max_multiplex=PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL );

        if (!TESTS_QUIET) {
	   printf("Maximum multiplexed counters=%d\n",max_multiplex);
	}

	if (!TESTS_QUIET) {
	   printf("Trying to multiplex as many as possible:\n");
	}

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__, "PAPI_assign_eventset_component",
		     retval );
	}

	retval = PAPI_set_multiplex( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__, "PAPI_create_multiplex", retval );
	}


	i = 0 | PAPI_PRESET_MASK;
	PAPI_enum_event( &i, PAPI_ENUM_FIRST );
	do {
	  retval = PAPI_get_event_info( i, &info );
	  if (retval==PAPI_OK) {
	     if (!TESTS_QUIET) printf("Adding %s: ",info.symbol);
	  }

	  retval = PAPI_add_event( EventSet, info.event_code );
	  if (retval!=PAPI_OK) {
	     if (!TESTS_QUIET) printf("Fail!\n");
	  }
	  else {
	     if (!TESTS_QUIET) printf("Success!\n");
	     added++;
	  }
	  events_tried++;

	} while (PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK );

	PAPI_shutdown(  );

	if (!TESTS_QUIET) {
           printf("Added %d of theoretical max %d\n",added,max_multiplex);
	}

	if (events_tried<max_multiplex) {
	   if (!TESTS_QUIET) {
              printf("Ran out of events before we ran out of room\n");
	   }
	}
	else if (added!=max_multiplex) {
	   test_fail(__FILE__, __LINE__,
		     "Couldn't max out multiplexed events", 1);
	}

	test_pass( __FILE__ );

	return 0;

}

