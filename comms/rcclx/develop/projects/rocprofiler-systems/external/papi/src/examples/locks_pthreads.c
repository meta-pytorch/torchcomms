/****************************************************************************
 * This program shows how to use PAPI_register_thread, PAPI_lock,           *
 * PAPI_unlock, PAPI_set_thr_specific, PAPI_get_thr_specific.               *
 * Warning: Don't use PAPI_lock and PAPI_unlock on platforms on which the   *
 * locking mechanisms are not implemented.                                  *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

#define LOOPS  		100000
#define SLEEP_VALUE	20000

int count;
int rank;

void *Master(void *arg)
{
   int i, retval, tmp;
   int *pointer, * pointer2;

   tmp = 20;
   pointer = &tmp;

   /* register the thread */
   if ( (retval=PAPI_register_thread())!= PAPI_OK )
      ERROR_RETURN(retval);
   
   /* save the pointer for late use */
   if ( (retval=PAPI_set_thr_specific(1,pointer))!= PAPI_OK )
      ERROR_RETURN(retval);
   /* change the value of tmp */
   tmp = 15;

   usleep(SLEEP_VALUE);
   PAPI_lock(PAPI_USR1_LOCK);
   /* Make sure Slaves are not sleeping */
   for (i = 0; i < LOOPS; i++) {
      count = 2 * count - i;
   }
   PAPI_unlock(PAPI_USR1_LOCK);

   /* retrieve the pointer saved by PAPI_set_thr_specific */
   if ( (retval=PAPI_get_thr_specific(1, (void *)&pointer2)) != PAPI_OK )
      ERROR_RETURN(retval);

   /* the output value should be 15 */
   printf("Thread specific data is %d \n", *pointer2);
   
   pthread_exit(NULL);
}

void *Slave(void *arg)
{
   int i;

   PAPI_lock(PAPI_USR2_LOCK);
   PAPI_lock(PAPI_USR1_LOCK);
   for (i = 0; i < LOOPS; i++) {
      count += i;
   }
   PAPI_unlock(PAPI_USR1_LOCK);
   PAPI_unlock(PAPI_USR2_LOCK);
   pthread_exit(NULL);
}



int main(int argc, char **argv)
{
   pthread_t master;
   pthread_t slave1;
   int result_m, result_s, rc, i;
   int retval;

   /* Setup a random number so compilers can't optimize it out */
   count = rand();
   result_m = count;
   rank = 0;

   for (i = 0; i < LOOPS; i++) {
      result_m = 2 * result_m - i;
   }
   result_s = result_m;

   for (i = 0; i < LOOPS; i++) {
      result_s += i;
   }

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
   {
      printf("Library initialization error! \n");
      exit(-1);
   }

   if ((retval = PAPI_thread_init(&pthread_self)) != PAPI_OK)
	 ERROR_RETURN(retval);

   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
      ERROR_RETURN(retval);

   PAPI_lock(PAPI_USR2_LOCK);
   rc = pthread_create(&master, NULL, Master, NULL);
   if (rc) {
      retval = PAPI_ESYS;
      ERROR_RETURN(retval);
   }
   rc = pthread_create(&slave1, NULL, Slave, NULL);
   if (rc) {
      retval = PAPI_ESYS;
      ERROR_RETURN(retval);
   }
   pthread_join(master, NULL);
   printf("Master: Expected: %d  Recieved: %d\n", result_m, count);
   if (result_m != count)
      ERROR_RETURN(1);
   PAPI_unlock(PAPI_USR2_LOCK);

   pthread_join(slave1, NULL);
   printf("Slave: Expected: %d  Recieved: %d\n", result_s, count);

   if (result_s != count)
      ERROR_RETURN(1);

   exit(0);
}
