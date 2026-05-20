#include <stdio.h>
#include <string.h>

#include "papi.h"

#include "do_loops.h"
#include "papi_test.h"

static int total = 0;				   /* total overflows */

static long long values[2];
static long long rapl_values[2];
static long long old_rapl_values[2] = {0,0};
static int rapl_backward=0;
static long long before_time, after_time;

int EventSet2=PAPI_NULL;

int quiet=0;

void handler( int EventSet, void *address, 
	      long long overflow_vector, void *context ) {

	( void ) context;
	( void ) address;
	( void ) overflow_vector;

#if 0
	fprintf( stderr, "handler(%d ) Overflow at %p! bit=%#llx \n",
                         EventSet, address, overflow_vector );
#endif

	PAPI_read(EventSet,values);
	PAPI_read(EventSet2,rapl_values);
   after_time = PAPI_get_real_nsec();
   double elapsed_time=((double)(after_time-before_time))/1.0e9;	

	if (!quiet) printf("%15lld %15lld %18lld %15lld %.3fms\n",
      values[0],values[1], 
      rapl_values[0], rapl_values[1], elapsed_time*1000.);

	if ((rapl_values[0]<old_rapl_values[0]) ||
	    (rapl_values[1]<old_rapl_values[1])) {
	   if (!quiet) printf("RAPL decreased!\n");
	   rapl_backward=1;
	}
	old_rapl_values[0]=rapl_values[0];
	old_rapl_values[1]=rapl_values[1];

	total++;
}


void do_ints(int n,int quiet)
{
  int i,c=n;

  for(i=0;i<n;i++) {
     c+=c*i*n;
  }
  if (!quiet) printf("do_ints result: %d\n",c);
}



int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	long long values0[2],values1[2],values2[2];
	int num_flops = 30000000, retval;
	int mythreshold;
	char event_name1[PAPI_MAX_STR_LEN];
        int PAPI_event;
	int cid,numcmp,rapl_cid;
	const PAPI_component_info_t *cmpinfo = NULL;
	int i;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	  test_fail(__FILE__, __LINE__,"PAPI_library_init",retval);
	}

	numcmp = PAPI_num_components();

	for(cid=0; cid<numcmp; cid++) {

	  if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	    test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n", 0);
	  }

	  if (strstr(cmpinfo->name,"rapl")) {
	    rapl_cid=cid;
	    if (!TESTS_QUIET) printf("Found rapl component at cid %d\n",
				     rapl_cid);
	    if (cmpinfo->num_native_events==0) {
              test_skip(__FILE__,__LINE__,"No rapl events found",0);
	    }
	    break;
	  }
	}

	/* Component not found */
	if (cid==numcmp) {
	  test_skip(__FILE__,__LINE__,"No rapl component found\n",0);
	}


	/* add PAPI_TOT_CYC and PAPI_TOT_INS to EventSet */
	retval=PAPI_create_eventset(&EventSet);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_create_eventset",retval);
	}

	retval=PAPI_add_event(EventSet,PAPI_TOT_CYC);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_add_event",retval);
	}

	retval=PAPI_add_event(EventSet,PAPI_TOT_INS);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_add_event",retval);
	}

	/* Add some RAPL events */
	retval=PAPI_create_eventset(&EventSet2);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_create_eventset",retval);
	}

	/* Add to eventSet2 for each package 0-n  */
   /* We use PACKAGE_ENERGY_CNT because it is an integer counter. */
   char raplEventBase[]="rapl:::PACKAGE_ENERGY_CNT:PACKAGE";
	i = 0;
	do {
		char buffer[80];
		sprintf(buffer, "%s%d", raplEventBase, i);
		retval=PAPI_add_named_event(EventSet2,buffer);
		++i;
	/* protect against insane PAPI library, the value 64 is the same value as 
     * RAPL_MAX_COUNTERS in linux-rapl.c, and feels reasonable. */
	} while ( 0 < retval && i < 64 );

	PAPI_event=PAPI_TOT_CYC;

	/* arbitrary, period of reporting, in total cycles. */
   /* our test routine is ~16 cycles per flop, get ~50 reports. */
	mythreshold = (num_flops/50)<<4;
	if (!TESTS_QUIET) {
	   printf("Using %#x for the overflow event on PAPI_TOT_CYC, threshold %d\n",
		  PAPI_event,mythreshold);
	}

	/* Start the run calibration run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}

	do_ints(num_flops,TESTS_QUIET);
	do_flops( 3000000 );

	/* stop the calibration run */
	retval = PAPI_stop( EventSet, values0 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}

   /* report header */
   if (!TESTS_QUIET) {
      printf("%15s %15s %18s %15s Elapsed\n", "PAPI_TOT_CYC", "PAPI_TOT_INS", 
         "PACKAGE_ENERGY_CNT", "--UNUSED--");
   }

	/* set up overflow handler */
	retval = PAPI_overflow( EventSet,PAPI_event,mythreshold, 0, handler );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_overflow",retval);
	}

	/* Start overflow run */
   before_time = PAPI_get_real_nsec();
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}
	retval = PAPI_start( EventSet2 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}

	do_ints(num_flops,TESTS_QUIET);
	do_flops( num_flops );

	/* stop overflow run */
	retval = PAPI_stop( EventSet, values1 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}

	retval = PAPI_stop( EventSet2, values2 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}

	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_overflow",retval);
	}

	retval = PAPI_event_code_to_name( PAPI_event, event_name1 );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__,"PAPI_event_code_to_name\n", retval);
	}

	if (!TESTS_QUIET) {
	   printf("%s: %lld(Calibration) %lld(OverflowRun)\n",event_name1,values0[0],values1[0]);
	}

	retval = PAPI_event_code_to_name( PAPI_TOT_INS, event_name1 );
	if (retval != PAPI_OK) {
	  test_fail(__FILE__, __LINE__,"PAPI_event_code_to_name\n",retval);
	}

	if (!TESTS_QUIET) {
	   printf("%s: %lld(Calibration) %lld(OverflowRun)\n",event_name1,values0[1],values1[1]);
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK ) {
	  test_fail(__FILE__, __LINE__,"PAPI_cleanup_eventset",retval);
	}

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_destroy_eventset",retval);
	}

	if (rapl_backward) {
	   test_fail(__FILE__, __LINE__,"RAPL counts went backward!",0);
	}

	test_pass( __FILE__ );

	return 0;
}
