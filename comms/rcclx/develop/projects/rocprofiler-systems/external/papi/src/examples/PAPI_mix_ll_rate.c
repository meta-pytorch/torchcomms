/*****************************************************************************
 * This example compares the measurement of IPC using the rate function      *
 * PAPI_ipc and the low-level API. Both methods should deliver the same      *
 * result for IPC.                                                           *
 * Note: There is no need to initialize PAPI for the low-level functions     *
 * since this is done by PAPI_ipc.                                           *
 *****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define THRESHOLD 10000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int your_slow_code();

int main()
{ 
  float real_time, proc_time, ipc;
  long long ins;
  int retval;
  int EventSet = PAPI_NULL;
  long_long values[2];

  if ( (retval = PAPI_ipc(&real_time, &proc_time, &ins ,&ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  your_slow_code();

  if ( (retval = PAPI_ipc( &real_time, &proc_time, &ins, &ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  printf("Results from PAPI_ipc:\n");
  printf("Real_time: %f Proc_time: %f Instructions: %lld IPC: %f\n", 
         real_time, proc_time,ins,ipc);

  if ( (retval = PAPI_rate_stop()) < PAPI_OK )
    ERROR_RETURN(retval);

  /* get IPC using low-level API */
  if ( (retval = PAPI_create_eventset(&EventSet)) < PAPI_OK )
    ERROR_RETURN(retval);
  
  if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) < PAPI_OK )
    ERROR_RETURN(retval);
  if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) < PAPI_OK )
    ERROR_RETURN(retval);

  if ( (retval = PAPI_start(EventSet)) < PAPI_OK )
    ERROR_RETURN(retval);

    your_slow_code();

  if ( (retval = PAPI_stop(EventSet, values)) < PAPI_OK )
    ERROR_RETURN(retval);
  
  ipc = (float) ((float)values[0] / (float) ( values[1]));

  printf("Results from the low-level API:\n");
  printf("IPC: %f\n", ipc);

  exit(0);
}

int your_slow_code()
{
  int i;
  double  tmp=1.1;

  for(i=1; i<2000; i++)
  { 
    tmp=(tmp+100)/i;
  }
  return 0;
}

