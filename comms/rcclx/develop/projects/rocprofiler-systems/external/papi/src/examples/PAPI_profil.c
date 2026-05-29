/****************************************************************************
 * PAPI_profil - generate PC histogram data                                 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define FLOPS 1000000
#define THRESHOLD 100000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int code_to_monitor()
{
   int i;
   double tmp=1.1;

   for(i=0; i < FLOPS; i++)
   {
      tmp=i+tmp;
      tmp++;
   }
   i = (int) tmp;
   return i;
}

int main()
{

    unsigned long length;
    vptr_t start, end;
    PAPI_sprofil_t * prof; 
    int EventSet = PAPI_NULL;
    /*must be initialized to PAPI_NULL before calling PAPI_create_event*/
    int PAPI_event,i,tmp = 0;
    char event_name[PAPI_MAX_STR_LEN];
    /*These are going to be used as buffers */
    unsigned short *profbuf;
    long long values[2];
    const PAPI_exe_info_t *prginfo = NULL;


    int retval;
    

   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/
    
   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      printf("Library initialization error! \n");
      exit(1);
   }
    

   if ((prginfo = PAPI_get_executable_info()) == NULL)
   {  
      fprintf(stderr, "Error in get executable information \n");
      exit(1);
   }

   start = prginfo->address_info.text_start;
   end = prginfo->address_info.text_end;
   length = (end - start);

   /* for PAPI_PROFIL_BUCKET_16 and scale = 65536, 
      profile buffer length == program address length.
      Larger bucket sizes would increase the buffer length.
      Smaller scale factors would decrease it.
      Handle with care...
   */
   profbuf = (unsigned short *)malloc(length);
   if (profbuf == NULL)
   {
      fprintf(stderr, "Not enough memory \n");
      exit(1);
   }
   memset(profbuf,0x00,length);

   /* Creating the eventset */
   if ( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   PAPI_event = PAPI_TOT_INS;
   /* Add Total Instructions Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_event)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Cycles Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* enable the collection of profiling information */
   if ((retval = PAPI_profil(profbuf, length, start, 65536, EventSet,
            PAPI_event, THRESHOLD, PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_16)) != PAPI_OK)
      ERROR_RETURN(retval);
   
   /* let's rock and roll */
   if ((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   code_to_monitor();
  
   if ((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* disable the collection of profiling information by setting threshold
      to 0
   */
   if ((retval = PAPI_profil(profbuf, length, start, 65536, EventSet,
            PAPI_event, 0, PAPI_PROFIL_POSIX)) != PAPI_OK)
      ERROR_RETURN(retval);
   
   printf("-----------------------------------------------------------\n");
   printf("Text start: %p, Text end: %p, \n",   
            prginfo->address_info.text_start,prginfo->address_info.text_end);
   printf("Data start: %p, Data end: %p\n",
            prginfo->address_info.data_start,prginfo->address_info.data_end);
   printf("BSS start : %p, BSS end: %p\n",
            prginfo->address_info.bss_start,prginfo->address_info.bss_end);
    
   printf("------------------------------------------\n");
        
   printf("Test type   : \tPAPI_PROFIL_POSIX\n");
   printf("------------------------------------------\n\n\n");  
   printf("PAPI_profil() hash table.\n");
   printf("address\t\tflat   \n");
   for (i = 0; i < (int) length/2; i++) 
   {
      if (profbuf[i]) 
         printf("%#lx\t%d \n",
               (unsigned long) start + (unsigned long) (2 * i), profbuf[i]);
   }

   printf("-----------------------------------------\n");

   retval = 0;
   for (i = 0; i < (int) length/2; i++)
      retval = retval || (profbuf[i]); 
   if (retval)
      printf("Test succeeds! \n");
   else
      printf( "No information in buffers\n");
   /* clean up */
   PAPI_shutdown();

   exit(0);
}


