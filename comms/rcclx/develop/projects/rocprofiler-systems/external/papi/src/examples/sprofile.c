/* This program shows how to use PAPI_sprofil */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include "papi.h" /* This needs to be included every time you use PAPI */

#define NUM_FLOPS  20000000
#define NUM_ITERS 100000
#define THRESHOLD 100000
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

#if (defined(linux) && defined(__ia64__)) || (defined(_AIX))
#define DO_FLOPS1 (vptr_t)(*(void **)do_flops1)
#define DO_FLOPS2 (vptr_t)(*(void **)do_flops2)
#else
#define DO_FLOPS1 (vptr_t)(do_flops1)
#define DO_FLOPS2 (vptr_t)(do_flops2)
#endif

void do_flops2(int);
volatile double t1 = 0.8, t2 = 0.9;
void do_flops1(int n)
{
   int i;
   double c = 22222.11;

   for (i = 0; i < n; i++)
      c -= t1 * t2;
}

void do_both(int n)
{
   int i;
   const int flops2 = NUM_FLOPS / n;
   const int flops1 = NUM_FLOPS / n;

   for (i = 0; i < n; i++) 
   {
      do_flops1(flops1);
      do_flops2(flops2);
   }
}

int main(int argc, char **argv)
{
   int i , PAPI_event;
   int EventSet = PAPI_NULL;
   unsigned short *profbuf;
   unsigned short *profbuf2;
   unsigned short *profbuf3;
   unsigned long length;
   vptr_t start, end;
   long long values[2];
   const PAPI_exe_info_t *prginfo = NULL;
   PAPI_sprofil_t sprof[3];
   int retval;

   /* initializaion */
   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
   {
      printf("Library initialization error! \n");
      exit(1);
   }

   if ((prginfo = PAPI_get_executable_info()) == NULL) 
      ERROR_RETURN(1);

   start = prginfo->address_info.text_start;
   end = prginfo->address_info.text_end;
   length = (end - start)/sizeof(unsigned short) * sizeof(unsigned short);
   printf("start= %p  end =%p \n", start, end);

   profbuf = (unsigned short *) malloc(length);
   if (profbuf == NULL) 
      ERROR_RETURN(PAPI_ESYS);

   memset(profbuf, 0x00, length );

   profbuf2 = (unsigned short *) malloc(length);
   if (profbuf2 == NULL) 
      ERROR_RETURN(PAPI_ESYS);

   memset(profbuf2, 0x00, length );

   profbuf3 = (unsigned short *) malloc(1 * sizeof(unsigned short));
   if (profbuf3 == NULL) 
      ERROR_RETURN(PAPI_ESYS);

   memset(profbuf3, 0x00, 1 * sizeof(unsigned short));

   /* First half */
   sprof[0].pr_base = profbuf;
   sprof[0].pr_size = length / 2;
   sprof[0].pr_off = DO_FLOPS2;
   fprintf(stderr, "do_flops is at %p %lx\n", &do_flops2, strtoul(sprof[0].pr_off,NULL,0));

   sprof[0].pr_scale = 65536;  /* constant needed by PAPI_sprofil */
   /* Second half */
   sprof[1].pr_base = profbuf2;
   sprof[1].pr_size = length / 2;
   sprof[1].pr_off = DO_FLOPS1;
   fprintf(stderr, "do_flops1 is at %p %lx\n", &do_flops1, strtoul(sprof[1].pr_off,NULL,0));
   sprof[1].pr_scale = 65536; /* constant needed by PAPI_sprofil */

   /* Overflow bin */
   sprof[2].pr_base = profbuf3;
   sprof[2].pr_size = 1;
   sprof[2].pr_off = 0;
   sprof[2].pr_scale = 0x2;  /* constant needed by PAPI_sprofil */

   /* Creating the eventset */
   if ( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   PAPI_event = PAPI_TOT_CYC;
   /* Add Total Instructions Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_event)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* set profile flag */
   if ((retval = PAPI_sprofil(sprof, 3, EventSet, PAPI_event, THRESHOLD,
                              PAPI_PROFIL_POSIX)) != PAPI_OK)
      ERROR_RETURN(retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   do_both(NUM_ITERS);

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* to clear the profile flag before removing the events */
   if ((retval = PAPI_sprofil(sprof, 3, EventSet, PAPI_event, 0,
                              PAPI_PROFIL_POSIX)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* free the resources hold by PAPI */
   PAPI_shutdown();

   printf("Test case: PAPI_sprofil()\n");
   printf("---------Buffer 1--------\n");
   for (i = 0; i < length / 2; i++) 
   {
      if (profbuf[i])
	printf("%#lx\t%d\n", strtoul(DO_FLOPS2,NULL,0) + 2 * i, profbuf[i]);
   }
   printf("---------Buffer 2--------\n");
   for (i = 0; i < length / 2; i++) 
   {
      if (profbuf2[i])
	printf("%#lx\t%d\n", strtoul(DO_FLOPS1,NULL,0) + 2 * i, profbuf2[i]);
   }
   printf("-------------------------\n");
   printf("%u samples fell outside the regions.\n", *profbuf3);

   exit(0);
}

/* Declare a and b to be volatile.
   This is to try to keep the
   compiler from optimizing the loop */
volatile double a = 0.5, b = 2.2;
void do_flops2(int n)
{
   int i;
   double c = 0.11;

   for (i = 0; i < n; i++) 
      c += a * b;
}

