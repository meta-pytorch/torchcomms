/* This file performs the following test: overflow dispatch with pthreads

   - This example  tests the dispatch of overflow calls from PAPI. The event
     set is counted in the default counting domain and default granularity, 
     depending on the platform. Usually this is the user domain 
    (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_TOT_INS (overflow monitor)
     + PAPI_TOT_CYC
   
   Each thread will do the followings :
   - enable overflow
   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - disable overflow
*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "papi.h"

#define THRESHOLD 200000
#define OVER_FMT    "handler(%d ) Overflow at %p! bit=%#llx \n"
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }


int total = 0;

void do_flops(int n)
{
    int i;
    double c = 0.11;
    double a = 0.5;
    double b = 6.2;

    for (i=0; i < n; i++) 
        c += a * b;
}

/* overflow handler */
void 
handler(int EventSet, void *address, long long overflow_vector, void *context)
{
   fprintf(stderr, OVER_FMT, EventSet, address, overflow_vector);
   total++;
}

void *Thread(void *arg)
{
	int retval;
	int EventSet1=PAPI_NULL;
	long long values[2];
	long long elapsed_us, elapsed_cyc;
  
	fprintf(stderr,"Thread %lx running PAPI\n",PAPI_thread_id());

	/* create the event set */
	if ( (retval = PAPI_create_eventset(&EventSet1))!=PAPI_OK)
    	ERROR_RETURN(retval);

 	/* query whether the event exists */
	if ((retval=PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK) 
    	ERROR_RETURN(retval);
	if ((retval=PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK) 
    	ERROR_RETURN(retval);

	/* add events to the event set */
	if ( (retval = PAPI_add_event(EventSet1, PAPI_TOT_INS))!= PAPI_OK)
    	ERROR_RETURN(retval);

	if ( (retval = PAPI_add_event(EventSet1, PAPI_TOT_CYC)) != PAPI_OK)
    	ERROR_RETURN(retval);

	elapsed_us = PAPI_get_real_usec();

	elapsed_cyc = PAPI_get_real_cyc();

	retval = PAPI_overflow(EventSet1, PAPI_TOT_CYC, THRESHOLD, 0, handler);
	if(retval !=PAPI_OK)
    	ERROR_RETURN(retval);

    /* start counting */
  	if((retval = PAPI_start(EventSet1))!=PAPI_OK)
    	ERROR_RETURN(retval);

    do_flops(*(int *)arg);
  
    if ((retval = PAPI_stop(EventSet1, values))!=PAPI_OK)
    	ERROR_RETURN(retval);

    elapsed_us = PAPI_get_real_usec() - elapsed_us;

    elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

    /* disable overflowing */
	retval = PAPI_overflow(EventSet1, PAPI_TOT_CYC, 0, 0, handler);
	if(retval !=PAPI_OK)
    	ERROR_RETURN(retval);

	/* remove the event from the eventset */
	retval = PAPI_remove_event(EventSet1, PAPI_TOT_INS);
    if (retval != PAPI_OK)
    	ERROR_RETURN(retval);

    retval = PAPI_remove_event(EventSet1, PAPI_TOT_CYC);
    if (retval != PAPI_OK)
    	ERROR_RETURN(retval);

    printf("Thread %#x PAPI_TOT_INS : \t%lld\n",(int)PAPI_thread_id(),
	 values[0]);
    printf("            PAPI_TOT_CYC: \t%lld\n", values[1]);
    printf("            Real usec   : \t%lld\n", elapsed_us);
    printf("            Real cycles : \t%lld\n", elapsed_cyc);

    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    pthread_t thread_one;
    pthread_t thread_two;
    int flops1, flops2;
    int rc,retval;
    pthread_attr_t attr;
    long long elapsed_us, elapsed_cyc;


	/* papi library initialization */
    if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	{
    	printf("Library initialization error! \n");
    	exit(1);
	}

	/* thread initialization */
    retval=PAPI_thread_init((unsigned long(*)(void))(pthread_self));
    if (retval != PAPI_OK)
    	ERROR_RETURN(retval);

	/* return the number of microseconds since some arbitrary starting point */
    elapsed_us = PAPI_get_real_usec();

	/* return the number of cycles since some arbitrary starting point */
    elapsed_cyc = PAPI_get_real_cyc();

	/* pthread attribution init */
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

	/* create the first thread */
    flops1 = 1000000;
    rc = pthread_create(&thread_one, &attr, Thread, (void *)&flops1);
    if (rc)
        ERROR_RETURN(rc);

	/* create the second thread */
    flops2 = 4000000;
    rc = pthread_create(&thread_two, &attr, Thread, (void *)&flops2);
    if (rc)
        ERROR_RETURN(rc);

	/* wait for the threads to finish */
    pthread_attr_destroy(&attr);
    pthread_join(thread_one, NULL); 
    pthread_join(thread_two, NULL);

	/* compute the elapsed cycles and microseconds */
    elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

    elapsed_us = PAPI_get_real_usec() - elapsed_us;

    printf("Master real usec   : \t%lld\n", elapsed_us);
    printf("Master real cycles : \t%lld\n", elapsed_cyc);

    /* clean up */
    PAPI_shutdown();
    exit(0);
}

