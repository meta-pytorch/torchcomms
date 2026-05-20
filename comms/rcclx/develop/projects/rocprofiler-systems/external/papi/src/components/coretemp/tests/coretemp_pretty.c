/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Vince Weaver
 *
 * test case that displays "pretty" coretemp output
 *
 * @brief
 *   Shows "pretty" coretemp output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 1

int main (int argc, char **argv)
{

    int retval,cid,coretemp_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    int code;
    char event_name[PAPI_MAX_STR_LEN];
    int r;
    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;
    double temperature;

        /* Set TESTS_QUIET variable */
     tests_quiet( argc, argv );

	/* PAPI Initialization */
     retval = PAPI_library_init( PAPI_VER_CURRENT );
     if ( retval != PAPI_VER_CURRENT ) {
	test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
     }

     if (!TESTS_QUIET) {
	printf("Trying all coretemp events\n");
     }

     numcmp = PAPI_num_components();

     for(cid=0; cid<numcmp; cid++) {

	if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	   test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n", 0);
	}

	if (strstr(cmpinfo->name,"coretemp")) {
	   coretemp_cid=cid;
	   if (!TESTS_QUIET) printf("Found coretemp component at cid %d\n",
				    coretemp_cid);
	   if (cmpinfo->disabled) {
	       if (!TESTS_QUIET) fprintf(stderr,"Coretemp component disabled: %s\n",
		       cmpinfo->disabled_reason);
	       test_skip(__FILE__, __LINE__,
		       "Component disabled\n", 0);
	   }
           if (cmpinfo->num_native_events==0) {
              test_skip(__FILE__,__LINE__,"No coretemp events found",0);
           }
	   break;
	}
     }




     code = PAPI_NATIVE_MASK;

     r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, coretemp_cid );

     while ( r == PAPI_OK ) {

        retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
	   printf("Error translating %#x\n",code);
	   test_fail( __FILE__, __LINE__,
                            "PAPI_event_code_to_name", retval );
	}

	retval = PAPI_get_event_info(code,&evinfo);
	if (retval != PAPI_OK) {
	  test_fail( __FILE__, __LINE__,
             "Error getting event info\n",retval);
	}

	   /****************************/
	   /* Print Temperature Inputs */
	   /****************************/
	if (strstr(event_name,"temp")) {

	     /* Only print inputs */
	  if (strstr(event_name,"_input")) {

	     if (!TESTS_QUIET) printf("%s ",event_name);

             EventSet = PAPI_NULL;

	     retval = PAPI_create_eventset( &EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_create_eventset()",retval);
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

	     retval = PAPI_stop( EventSet, values);
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
	     }

	     temperature=(values[0]/1000.0);

             if (!TESTS_QUIET) printf("\tvalue: %.2lf %s\n",
				      temperature,
				      evinfo.long_descr
				      );

	     retval = PAPI_cleanup_eventset( EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_cleanup_eventset()",retval);
	     }

	     retval = PAPI_destroy_eventset( &EventSet );
             if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_destroy_eventset()",retval);
	     }
	  }
	}

	   /****************************/
	   /* Print Voltage Inputs */
	   /****************************/
	if (strstr(event_name,".in")) {

	     /* Only print inputs */
	  if (strstr(event_name,"_input")) {

	     if (!TESTS_QUIET) printf("%s ",event_name);

             EventSet = PAPI_NULL;

	     retval = PAPI_create_eventset( &EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_create_eventset()",retval);
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

	     retval = PAPI_stop( EventSet, values);
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
	     }

	     temperature=(values[0]/1000.0);

             if (!TESTS_QUIET) printf("\tvalue: %.2lf %s\n",
				      temperature,
				      evinfo.long_descr
				      );

	     retval = PAPI_cleanup_eventset( EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_cleanup_eventset()",retval);
	     }

	     retval = PAPI_destroy_eventset( &EventSet );
             if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_destroy_eventset()",retval);
	     }
	  }
	}
	/********************/
	/* Print Fan Inputs */
	/********************/
	else if (strstr(event_name,"fan")) {

	     /* Only print inputs */
	  if (strstr(event_name,"_input")) {

             if (!TESTS_QUIET) printf("%s ",event_name);

             EventSet = PAPI_NULL;

	     retval = PAPI_create_eventset( &EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_create_eventset()",retval);
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

	     retval = PAPI_stop( EventSet, values);
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
	     }

             if (!TESTS_QUIET) printf("\tvalue: %lld %s\n",values[0],
				      evinfo.long_descr);

	     retval = PAPI_cleanup_eventset( EventSet );
	     if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_cleanup_eventset()",retval);
	     }

	     retval = PAPI_destroy_eventset( &EventSet );
             if (retval != PAPI_OK) {
	        test_fail(__FILE__, __LINE__,
                              "PAPI_destroy_eventset()",retval);
	     }
	  }

	}
	else {
	  /* Skip unknown */
	}

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, coretemp_cid );
     }

	test_pass( __FILE__ );

	return 0;
}

