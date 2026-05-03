/**
 * @author  Vince Weaver
 *
 * test case for mx myrinet component
 *
 *
 * @brief
 *   Tests basic mx myrinet functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 3

int main (int argc, char **argv)
{

	int retval,cid,numcmp,our_cmp;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
	int code;
	const PAPI_component_info_t *cmpinfo = NULL;
	int quiet=0;

        /* Set quiet variable */
        quiet=tests_quiet( argc, argv );

	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
	}

	if (!quiet) {
	   printf("Trying mutiple reads in MX component\n");
	}

        numcmp = PAPI_num_components();
	our_cmp=-1;

	for(cid=0; cid<numcmp; cid++) {

	   if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	      test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n", 0);
	   }

	   if (strstr(cmpinfo->name,"mx")) {
	     if (!quiet) printf("\tFound Myrinet component %d - %s\n", cid, cmpinfo->name);
	     our_cmp=cid;
	     break;
	   }

	}

	if (our_cmp<0) {
	   test_skip(__FILE__, __LINE__,"MX component not found\n", 0);
	}

	if (cmpinfo->num_native_events<=0) {
	   test_skip(__FILE__, __LINE__,"MX component not found\n", 0);
	}
  
     
	EventSet = PAPI_NULL;

	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, 
                              "PAPI_create_eventset()",retval);
	}

	retval=PAPI_event_name_to_code("mx:::COUNTERS_UPTIME",&code);
	if (retval!=PAPI_OK) {
           test_fail(__FILE__, __LINE__, 
                              "could not add event COUNTERS_UPTIME",retval);
	}

	retval = PAPI_add_event( EventSet, code );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, 
                                 "PAPI_add_event()",retval);
	}

	retval=PAPI_event_name_to_code("mx:::PUSH_OBSOLETE",&code);
	if (retval!=PAPI_OK) {
           test_fail(__FILE__, __LINE__, 
                              "could not add event PUSH_OBSOLETE",retval);
	}

	retval = PAPI_add_event( EventSet, code );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, 
                                 "PAPI_add_event()",retval);
	}

	retval=PAPI_event_name_to_code("mx:::PKT_MISROUTED",&code);
	if (retval!=PAPI_OK) {
           test_fail(__FILE__, __LINE__, 
                              "could not add event PKT_MISROUTED",retval);
	}

	retval = PAPI_add_event( EventSet, code );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, 
                                 "PAPI_add_event()",retval);
	}

	retval = PAPI_start( EventSet);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
	}

	retval = PAPI_read( EventSet, values);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_read()",retval);
	}

	if (!quiet) printf("%lld %lld %lld\n",values[0],values[1],values[2]);

	retval = PAPI_stop( EventSet, values);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
	}

	if (!quiet) printf("%lld %lld %lld\n",values[0],values[1],values[2]);



	test_pass( __FILE__ );

	return 0;
}

