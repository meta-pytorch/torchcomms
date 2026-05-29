/* 
 * Test case for appio
 * Author: Tushar Mohan
 *         tusharmohan@gmail.com
 * 
 * Description: This test case reads from standard linux /etc files in
 *              four separate threads and copies the output to /dev/null
 *              READ and WRITE statistics for each of the threads is
 *              summarized at the end.
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 6
const char* names[NUM_EVENTS] = {"READ_CALLS", "READ_BYTES","READ_USEC","WRITE_CALLS","WRITE_BYTES","WRITE_USEC"};

#define NUM_INFILES 4
static const char* files[NUM_INFILES] = {"/etc/passwd", "/etc/group", "/etc/protocols", "/etc/nsswitch.conf"};

void *ThreadIO(void *arg) {
  unsigned long tid = (unsigned long)pthread_self();
  if (!TESTS_QUIET) printf("\nThread %#lx: will read %s and write it to /dev/null\n", tid,(const char*) arg);
  int EventSet = PAPI_NULL;
  long long values[NUM_EVENTS];
  int retval;
  int e;
  int event_code;

  /* Create the Event Set */
  if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
    fprintf(stderr, "Error creating event set\n");
      exit(2);
  }

  for (e=0; e<NUM_EVENTS; e++) {
    retval = PAPI_event_name_to_code((char*)names[e], &event_code);
    if (retval != PAPI_OK) {
      fprintf(stderr, "Error getting code for %s\n", names[e]);
      exit(2);
    }
    retval = PAPI_add_event(EventSet, event_code);
    if (retval != PAPI_OK) {
      fprintf(stderr, "Error adding %s to event set\n", names[e]);
      exit(2);
    }
  }

  /* Start counting events */
  if (PAPI_start(EventSet) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_start\n");
    exit(1);
  }
 
//if (PAPI_read_counters(EventSet, values) != PAPI_OK)
//   handle_error(1);
//printf("After reading the counters: %lld\n",values[0]);

  int fdin = open((const char*)arg, O_RDONLY);
  if (fdin < 0) perror("Could not open file for reading: \n");

  int bytes = 0;
  char buf[1024];

  int fdout = open("/dev/null", O_WRONLY);
  if (fdout < 0) perror("Could not open /dev/null for writing: \n");
  while ((bytes = read(fdin, buf, 1024)) > 0) {
    write(fdout, buf, bytes);
  }
  close(fdout);

  /* Stop counting events */
  if (PAPI_stop(EventSet, values) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop\n");
  }

  if (!TESTS_QUIET) {
    for (e=0; e<NUM_EVENTS; e++)  
      printf("Thread %#lx: %s: %lld\n", tid, names[e], values[e]);
  }
  return(NULL);
}

int main(int argc, char** argv) {
  pthread_t *callThd;
  int i, numthrds;
  int retval;
  pthread_attr_t attr;

  /* Set TESTS_QUIET variable */
  tests_quiet( argc, argv );

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }


  pthread_attr_init(&attr);
  if (PAPI_thread_init(pthread_self) != PAPI_OK) {
    fprintf(stderr, "PAPI_thread_init returned an error\n");
    exit(1);
  }
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  retval = pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
  if (retval != 0) {
    fprintf(stderr,"This system does not support kernel scheduled pthreads.\n");
    exit(1);
  }
#endif

  numthrds = NUM_INFILES;
  if (!TESTS_QUIET) printf("%d threads\n",numthrds);
  callThd = (pthread_t *)malloc(numthrds*sizeof(pthread_t));

  int rc ;
  for (i=0;i<(numthrds-1);i++) {
    rc = pthread_create(callThd+i, &attr, ThreadIO, (void *) files[i]);
    if (rc != 0) perror("Error creating thread using pthread_create()");
  }
  ThreadIO((void *)files[numthrds-1]);
  pthread_attr_destroy(&attr);

  for (i=0;i<(numthrds-1);i++)
    pthread_join(callThd[i], NULL);

  test_pass( __FILE__ );
  return 0;
}
