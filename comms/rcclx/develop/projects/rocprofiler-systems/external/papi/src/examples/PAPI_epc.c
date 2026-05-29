/*****************************************************************************
 * This example demonstrates the usage of the function PAPI_epc which        *
 * measures arbitrary events per cpu cycle                                   *
 *****************************************************************************/

/*****************************************************************************
 * The first call to PAPI_epc() will initialize the PAPI interface,          *
 * set up the counters to monitor the user specified event, PAPI_TOT_CYC,    *
 * and PAPI_REF_CYC (if it exists) and start the counters. Subsequent calls  *
 * will read the counters and return real time, process time, event counts,  *
 * the core and reference cycle count and EPC rate since the latest call to  *
 * PAPI_epc().                                                               *
 *****************************************************************************/


 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

int your_slow_code();

int main()
{ 
  float real_time, proc_time, epc;
  long long ref, core, evt;
  float real_time_i, proc_time_i, epc_i;
  long long ref_i, core_i, evt_i;
  int retval;

  if((retval=PAPI_epc(PAPI_TOT_INS, &real_time_i, &proc_time_i, &ref_i, &core_i, &evt_i, &epc_i)) < PAPI_OK)
  { 
    printf("Could not initialise PAPI_epc \n");
    printf("retval: %d\n", retval);
    exit(1);
  }

  your_slow_code();

  
  if((retval=PAPI_epc(PAPI_TOT_INS, &real_time, &proc_time, &ref, &core, &evt, &epc))<PAPI_OK)
  {    
    printf("retval: %d\n", retval);
    exit(1);
  }


  printf("Real_time: %f Proc_time: %f Ref_clock: %lld Core_clock: %lld Events: %lld EPC: %f\n",
         real_time, proc_time, ref, core, evt, epc);

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

