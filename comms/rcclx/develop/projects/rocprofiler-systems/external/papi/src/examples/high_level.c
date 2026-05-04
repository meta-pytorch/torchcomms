/*****************************************************************************
*  This example code shows how to use PAPI's High level functions.           * 
*  Events to be recorded are determined via an environment variable          *
*  PAPI_EVENTS that lists comma separated events for any component.          *
*  If events are not specified via the environment variable PAPI_EVENTS, an  *
*  output with default events is generated after the run. If supported by    *
*  the respective machine the following default events are recorded:         *
*      perf::TASK-CLOCK                                                      *
*      PAPI_TOT_INS                                                          *
*      PAPI_TOT_CYC                                                          *
*      PAPI_FP_INS                                                           *
*      PAPI_FP_OPS or PAPI_DP_OPS or PAPI_SP_OPS                             *
******************************************************************************/ 

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define THRESHOLD 10000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

/* stupid codes to be monitored */ 
void computation_mult()
{
   double tmp=1.0;
   int i=1;
   for( i = 1; i < THRESHOLD; i++ )
   {
      tmp = tmp*i;
   }
}

/* stupid codes to be monitored */ 
void computation_add()
{
   int tmp = 0;
   int i=0;

   for( i = 0; i < THRESHOLD; i++ )
   {
      tmp = tmp + i;
   }

}


int main()
{
   int retval;
   char errstring[PAPI_MAX_STR_LEN];

   retval = PAPI_hl_region_begin("computation_add");
   if ( retval != PAPI_OK )
      ERROR_RETURN(retval);

   /* Your code goes here*/
   computation_add();

   retval = PAPI_hl_read("computation_add");
   if ( retval != PAPI_OK )
      ERROR_RETURN(retval);

   /* Your code goes here*/
   computation_add();

   retval = PAPI_hl_region_end("computation_add");
   if ( retval != PAPI_OK )
      ERROR_RETURN(retval);


   retval = PAPI_hl_region_begin("computation_mult");
   if ( retval != PAPI_OK )
      ERROR_RETURN(retval);
   
   /* Your code goes here*/
   computation_mult();

   retval = PAPI_hl_region_end("computation_mult");
   if ( retval != PAPI_OK )
      ERROR_RETURN(retval);

   exit(0);
}
