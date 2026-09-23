/** 
 * @author  Vince Weaver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"

#define MAX_EVENTS 128

char events[MAX_EVENTS][BUFSIZ];
char units[MAX_EVENTS][BUFSIZ];
int data_type[MAX_EVENTS];
char filenames[MAX_EVENTS][BUFSIZ];

FILE *fff[MAX_EVENTS];

static int num_events=0;

int main (int argc, char **argv)
{

	int retval,cid,rapl_cid=-1,numcmp;
	int EventSet = PAPI_NULL;
	long long values[MAX_EVENTS];
	int i,code,enum_retval;
	PAPI_event_info_t evinfo;
	const PAPI_component_info_t *cmpinfo = NULL;
	long long start_time,before_time,after_time;
	double elapsed_time,total_time;
	char event_name[BUFSIZ];

	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"PAPI_library_init failed\n");
		exit(1);
	}

	numcmp = PAPI_num_components();

	for(cid=0; cid<numcmp; cid++) {

		if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
			fprintf(stderr,"PAPI_get_component_info failed\n");
			exit(1);
		}

		if (strstr(cmpinfo->name,"rapl")) {
			rapl_cid=cid;
			printf("Found rapl component at cid %d\n", rapl_cid);

			if (cmpinfo->disabled) {
				fprintf(stderr,"No rapl events found: %s\n",
						cmpinfo->disabled_reason);
				exit(1);
			}
			break;
		}
	}

	/* Component not found */
	if (cid==numcmp) {
		fprintf(stderr,"No rapl component found\n");
		exit(1);
	}

	/* Find Events */
	code = PAPI_NATIVE_MASK;

	enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );

	while ( enum_retval == PAPI_OK ) {

		retval = PAPI_event_code_to_name( code, event_name );
		if ( retval != PAPI_OK ) {
			printf("Error translating %#x\n",code);
			exit(1);
		}

		printf("Found: %s\n",event_name);
		strncpy(events[num_events],event_name,BUFSIZ);
		sprintf(filenames[num_events],"results.%s",event_name);


        /* Find additional event information: unit, data type */
		retval = PAPI_get_event_info(code, &evinfo);
		if (retval != PAPI_OK) {
			printf("Error getting event info for %#x\n",code);
			exit(1);
		}

		strncpy(units[num_events],evinfo.units,sizeof(units[0])-1);
		/* buffer must be null terminated to safely use strstr operation on it below */
		units[num_events][sizeof(units[0])-1] = '\0';

		data_type[num_events] = evinfo.data_type;

		num_events++;

		if (num_events==MAX_EVENTS) {
			printf("Too many events! %d\n",num_events);
			exit(1);
		}

		enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );

	}



	if (num_events==0) {
		printf("Error!  No RAPL events found!\n");
		exit(1);
	}

	/* Open output files */
	for(i=0;i<num_events;i++) {
		fff[i]=fopen(filenames[i],"w");
		if (fff[i]==NULL) {
			fprintf(stderr,"Could not open %s\n",filenames[i]);
			exit(1);
		}
	}


	/* Create EventSet */
	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK) {
		fprintf(stderr,"Error creating eventset!\n");
	}

	for(i=0;i<num_events;i++) {

		retval = PAPI_add_named_event( EventSet, events[i] );
		if (retval != PAPI_OK) {
			fprintf(stderr,"Error adding event %s\n",events[i]);
		}
	}



	start_time=PAPI_get_real_nsec();

	while(1) {

		/* Start Counting */
		before_time=PAPI_get_real_nsec();
		retval = PAPI_start( EventSet);
		if (retval != PAPI_OK) {
			fprintf(stderr,"PAPI_start() failed\n");
			exit(1);
		}


		usleep(100000);

		/* Stop Counting */
		after_time=PAPI_get_real_nsec();
		retval = PAPI_stop( EventSet, values);
		if (retval != PAPI_OK) {
			fprintf(stderr, "PAPI_start() failed\n");
		}

		total_time=((double)(after_time-start_time))/1.0e9;
		elapsed_time=((double)(after_time-before_time))/1.0e9;

		for(i=0;i<num_events;i++) {


			if (!strstr(events[i],"ENERGY")) {

                /* Scaled fixed value */
				if (data_type[i] == PAPI_DATATYPE_FP64) {

					union {
						long long ll;
						double fp;
					} result;

					result.ll=values[i];


					fprintf(fff[i],"%.4f  %.3f  (* %s in (%s) *)\n",
							total_time,
							result.fp, 
							events[i], units[i] );

				} 
				/* Fixed value counts */
				else if (data_type[i] == PAPI_DATATYPE_UINT64) {

					fprintf(fff[i],"%.4f  %lld  (* %s *)\n",
							total_time,
							values[i], 
							events[i] );

				}
			}

			/* Energy measurement counts */
			else if (strstr(events[i],"ENERGY_CNT")) {  

				fprintf(fff[i],"%.4f  %lld  (* %s *)\n",
						total_time,
						values[i], 
						events[i] );

			}

			/* Scaled energy measurements */
			else {
				fprintf(fff[i],"%.4f  %.3f %s  %.3f %s  (* Average Power for %s *)\n",
						total_time,
						((double)values[i]/1.0e9), "J",
						((double)values[i]/1.0e9)/elapsed_time, "W", 
						events[i] );
			};

			fflush(fff[i]);
		}
	}

	return 0;
}

