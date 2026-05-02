/*****************************************************************************
 * This example demonstrates the usage of the function PAPI_ipc which        *
 * measures the number of instructions executed per cpu cycle                *
 *****************************************************************************/

/*****************************************************************************
 * The first call to PAPI_ipc initializes the PAPI library, set up the       *
 * counters to monitor PAPI_TOT_INS and PAPI_TOT_CYC events, and start the   *
 * counters. Subsequent calls will read the counters and return real time,   *
 * process time, instructions, and the instructions per cycle rate since the *
 * latest call to PAPI_ipc.                                                  *
 *****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

int your_slow_code();

int main()
{ 
  float real_time, proc_time,ipc;
  long long ins;
  float real_time_i, proc_time_i, ipc_i;
  long long ins_i;
  int retval;

  if((retval=PAPI_ipc(&real_time_i,&proc_time_i,&ins_i,&ipc_i)) < PAPI_OK)
  { 
    printf("Could not initialise PAPI_ipc \n");
    printf("retval: %d\n", retval);
    exit(1);
  }

  your_slow_code();

  
  if((retval=PAPI_ipc( &real_time, &proc_time, &ins, &ipc))<PAPI_OK)
  {    
    printf("retval: %d\n", retval);
    exit(1);
  }


  printf("Real_time: %f Proc_time: %f Instructions: %lld IPC: %f\n", 
         real_time, proc_time,ins,ipc);

  /* clean up */
  PAPI_shutdown();
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

