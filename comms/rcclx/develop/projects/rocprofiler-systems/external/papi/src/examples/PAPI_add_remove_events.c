/******************************************************************************
 * This is a simple low level function demonstration on using PAPI_add_events *
 * to add an array of events to a created eventset, we are going to use these *
 * events to monitor a set of instructions, start the counters, read the      *
 * counters and then cleanup the eventset when done. In this example we use   *
 * the presets PAPI_TOT_INS and PAPI_TOT_CYC. PAPI_add_events,PAPI_start,     *
 * PAPI_stop, PAPI_clean_eventset, PAPI_destroy_eventset and                  *
 * PAPI_create_eventset all return PAPI_OK(which is 0) when succesful.        *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define NUM_EVENT 2
#define THRESHOLD 100000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }


int main(){
 
	int i,retval,tmp;
    int EventSet = PAPI_NULL;
    /*must be initialized to PAPI_NULL before calling PAPI_create_event*/

    int event_codes[NUM_EVENT]={PAPI_TOT_INS,PAPI_TOT_CYC}; 
    char errstring[PAPI_MAX_STR_LEN];
    long long values[NUM_EVENT];

   /***************************************************************************
   * This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it *
   * is likely that PAPI won't work correctly.If there is an error, retval    *
   * keeps track of the version number.                                       *
   ****************************************************************************/

    if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
    {
		fprintf(stderr, "Error: %s\n", errstring);
    	exit(1);
    }

   
  	/* Creating event set   */
  	if ((retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)
		ERROR_RETURN(retval);


  	/* Add the array of events PAPI_TOT_INS and PAPI_TOT_CYC to the eventset*/
	if ((retval=PAPI_add_events(EventSet, event_codes, NUM_EVENT)) != PAPI_OK)
		ERROR_RETURN(retval);
  
       
	/* Start counting */
	if ( (retval=PAPI_start(EventSet)) != PAPI_OK)
		ERROR_RETURN(retval);
     
    /*** this is where your computation goes *********/
	for(i=0;i<1000;i++)
    {
    	tmp = tmp+i;
    }  

    /* Stop counting, this reads from the counter as well as stop it. */
    if ( (retval=PAPI_stop(EventSet,values)) != PAPI_OK)
		ERROR_RETURN(retval);

	printf("\nThe total instructions executed are %lld, total cycles %lld\n",
            values[0],values[1]);

    
	if ( (retval=PAPI_remove_events(EventSet,event_codes, NUM_EVENT))!=PAPI_OK)
		ERROR_RETURN(retval);

    /* Free all memory and data structures, EventSet must be empty. */
    if ( (retval=PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
		ERROR_RETURN(retval);

    /* free the resources used by PAPI */
    PAPI_shutdown();
   
    exit(0);
}
