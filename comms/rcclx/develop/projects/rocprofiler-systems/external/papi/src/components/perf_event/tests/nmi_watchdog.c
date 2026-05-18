/* If the NMI watchdog is enabled it will steal a performance counter.   */
/* There is a bug that if you try to use the maximum number of counters  */
/*   (not counting the stolen one) with a group leader, sys_perf_open()  */
/* will indicate success, as will starting the count, but you will fail  */
/* at read time.                                                         */

/* This bug still exists in 3.x */
/* The perf NMI watchdog was not introduced until 2.6.34        */

/* This also triggers in the case of the schedulability bug      */
/* but since that was fixed in 2.6.34 then in theory there is    */
/* no overlap in the tests.                                      */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"


static int detect_nmi_watchdog(void) {

	int watchdog_detected=0,watchdog_value=0;
	FILE *fff;

	fff=fopen("/proc/sys/kernel/nmi_watchdog","r");
	if (fff!=NULL) {
		if (fscanf(fff,"%d",&watchdog_value)==1) {
			if (watchdog_value>0) watchdog_detected=1;
		}
		fclose(fff);
	}
	else {
		watchdog_detected=-1;
	}

	return watchdog_detected;
}

int main( int argc, char **argv ) {

	int retval,watchdog_active=0;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	watchdog_active=detect_nmi_watchdog();

	if (watchdog_active<0) {
		test_skip( __FILE__, __LINE__,
			"nmi_watchdog file does not exist\n", 0);
	}

	if (watchdog_active) {
		if (!quiet) {
	printf("\nOn perf_event kernels with the nmi_watchdog enabled\n");
	printf("the watchdog steals an event, but the scheduability code\n");
	printf("is not notified.  Thus adding a full complement of events\n");
	printf("seems to pass, but then fails at read time.\n");
	printf("Because of this, PAPI has to do some slow workarounds.\n");
	printf("For best PAPI performance, you may wish to disable\n");
	printf("the watchdog by running (as root)\n");
	printf("\techo \"0\" > /proc/sys/kernel/nmi_watchdog\n\n");
		}

		test_warn( __FILE__, __LINE__,
			"NMI Watchdog Active, enabling slow workarounds", 0 );
	}

	test_pass( __FILE__ );

 	return 0;
}



