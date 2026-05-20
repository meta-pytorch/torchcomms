/****************************************************************************
 * This is a simple low level example for getting information on the system *
 * hardware. This function PAPI_get_hardware_info(), returns a pointer to a *
 * structure of type PAPI_hw_info_t, which contains number of CPUs, nodes,  *
 * vendor number/name for CPU, CPU revision, clock speed.                   *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

int main()
{
   const PAPI_hw_info_t *hwinfo = NULL;
   int retval;
  
   /*************************************************************************** 
   *  This part initializes the library and compares the version number of the*
   * header file, to the version of the library, if these don't match then it *
   * is likely that PAPI won't work correctly.If there is an error, retval    *
   * keeps track of the version number.                                       *
   ***************************************************************************/


   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      printf("Library initialization error! \n");
      exit(1);
   }
     
   /* Get hardware info*/      
   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
   {
      printf("PAPI_get_hardware_info error! \n");
      exit(1);
   }
   /* when there is an error, PAPI_get_hardware_info returns NULL */


   printf("%d CPU  at %f Mhz.\n",hwinfo->totalcpus,hwinfo->mhz);
   printf(" model string is %s \n", hwinfo->model_string);

   /* clean up */ 
   PAPI_shutdown();

   exit(0);

}

