/*****************************************************************************
 * This example compares the measurement of IPC using the rate function      *
 * PAPI_ipc and the high-level region instrumentation. Both methods should   *
 * deliver the same result for IPC.                                          *
 * Hint: Use PAPI's high-level output script to print the measurement report *
 * of the high-level API.                                                    *
 *                                                                           *
 * ../high-level/scripts/papi_hl_output_writer.py --type=accumulate          *
 *****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define THRESHOLD 10000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int your_slow_code();

int main()
{ 
  float real_time, proc_time,ipc;
  long long ins;
  int retval;

  if ( (retval = PAPI_ipc(&real_time, &proc_time, &ins ,&ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  your_slow_code();

  if ( (retval = PAPI_ipc( &real_time, &proc_time, &ins, &ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  printf("Real_time: %f Proc_time: %f Instructions: %lld IPC: %f\n", 
         real_time, proc_time,ins,ipc);


  if ( (retval = PAPI_hl_region_begin("slow_code")) < PAPI_OK )
    ERROR_RETURN(retval);

   your_slow_code();

  if ( (retval = PAPI_hl_region_end("slow_code")) < PAPI_OK )
    ERROR_RETURN(retval);


  if ( (retval = PAPI_ipc(&real_time, &proc_time, &ins ,&ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  your_slow_code();

  if ( (retval = PAPI_ipc( &real_time, &proc_time, &ins, &ipc)) < PAPI_OK )
    ERROR_RETURN(retval);

  printf("Real_time: %f Proc_time: %f Instructions: %lld IPC: %f\n", 
         real_time, proc_time,ins,ipc);

  if ( (retval = PAPI_rate_stop()) < PAPI_OK )
    ERROR_RETURN(retval);

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

