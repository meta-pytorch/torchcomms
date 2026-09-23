/*
 * File:    failed_events.c
 * Author:  Vince Weaver <vincent.weaver@maine.edu>
 */

/* This test tries adding events that don't exist */
/* We've had issues where the name resolution code might do weird */
/* things when passed invalid event names */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"


#define LARGE_NAME_SIZE	4096

char large_name[LARGE_NAME_SIZE];

int
main( int argc, char **argv )
{

	int i, k, err_count = 0;
	int retval;
	PAPI_event_info_t info, info1;
	const PAPI_component_info_t* cmpinfo;
	int numcmp, cid;
	int quiet;

	int EventSet = PAPI_NULL;

	/* Set quiet variable */
	quiet=tests_quiet( argc, argv );

	/* Init PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("Test adding invalid events.\n");
	}

	/* Create an eventset */
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}


	/* Simple Event */
	if (!quiet) {
		printf("+ Simple invalid event\t");
	}

	retval=PAPI_add_named_event(EventSet,"INVALID_EVENT");
	if (retval==PAPI_OK) {
		if (!quiet) {
			printf("Unexpectedly opened!\n");
			err_count++;
		}
	}
	else {
		if (!quiet) printf("OK\n");
	}

	/* Extra Colons */
	if (!quiet) {
		printf("+ Extra colons\t");
	}

	retval=PAPI_add_named_event(EventSet,"INV::::AL:ID:::_E=3V::E=NT");
	if (retval==PAPI_OK) {
		if (!quiet) {
			printf("Unexpectedly opened!\n");
			err_count++;
		}
	}
	else {
		if (!quiet) printf("OK\n");
	}


	/* Large Invalid Event */
	if (!quiet) {
		printf("+ Large invalid event\t");
	}

	memset(large_name,'A',LARGE_NAME_SIZE);
	large_name[LARGE_NAME_SIZE-1]=0;

	retval=PAPI_add_named_event(EventSet,large_name);
	if (retval==PAPI_OK) {
		if (!quiet) {
			printf("Unexpectedly opened!\n");
			err_count++;
		}
	}
	else {
		if (!quiet) printf("OK\n");
	}

	/* Large Unterminated Invalid Event */
	if (!quiet) {
		printf("+ Large unterminated invalid event\t");
	}

	memset(large_name,'A',LARGE_NAME_SIZE);

	retval=PAPI_add_named_event(EventSet,large_name);
	if (retval==PAPI_OK) {
		if (!quiet) {
			printf("Unexpectedly opened!\n");
			err_count++;
		}
	}
	else {
		if (!quiet) printf("OK\n");
	}


	/* Randomly modifying valid events */
	if (!quiet) {
		printf("+ Randomly modifying valid events\t");
	}

	numcmp = PAPI_num_components(  );

	/* Loop through all components */
	for( cid = 0; cid < numcmp; cid++ ) {


		cmpinfo = PAPI_get_component_info( cid );
		if (cmpinfo  == NULL) {
			test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 2 );
		}

		/* Include disabled components */
		if (cmpinfo->disabled) {
			// continue;
		}


		/* For platform independence, always ASK FOR the first event */
		/* Don't just assume it'll be the first numeric value */
		i = 0 | PAPI_NATIVE_MASK;
		retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );

		do {
			retval = PAPI_get_event_info( i, &info );

			  k = i;
	  if ( PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid )==PAPI_OK ) {
	     do {
		retval = PAPI_get_event_info( k, &info1 );



		/* Skip perf_raw event as it is hard to error out */
		if (strstr(info1.symbol,"perf_raw")) {
			break;
		}

//		printf("%s\n",info1.symbol);

		if (strlen(info1.symbol)>5) {
			info1.symbol[strlen(info1.symbol)-4]^=0xa5;

			retval=PAPI_add_named_event(EventSet,info1.symbol);
			if (retval==PAPI_OK) {
				if (!quiet) {
					printf("Unexpectedly opened %s!\n",
						info1.symbol);
					err_count++;
				}
			}
		}
	     } while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK );
	  } else {
	    /* Event didn't have any umasks */

      // PROBLEM: info1 is NOT initialized by anyone!
      // Original code referenced info1, changed to info. [Tony C. 11-27-19] 
//		printf("%s\n",info.symbol);
		if (strlen(info.symbol)>5) {
			info.symbol[strlen(info.symbol)-4]^=0xa5;

			retval=PAPI_add_named_event(EventSet,info.symbol);
			if (retval==PAPI_OK) {
				if (!quiet) {
					printf("Unexpectedly opened %s!\n",
						info.symbol);
					err_count++;
				}
			}
		}
	  }

       } while ( PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid ) == PAPI_OK );

    }



	if ( err_count ) {
		if (!quiet) {
			printf( "%d Invalid events added.\n", err_count );
		}
		test_fail( __FILE__, __LINE__, "Invalid events added", 1 );
	}

	test_pass( __FILE__ );

	return 0;
}
