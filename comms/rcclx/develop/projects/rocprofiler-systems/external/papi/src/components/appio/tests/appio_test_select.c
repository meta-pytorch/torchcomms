/* 
 * Test case for appio
 * Author: Tushar Mohan
 *         tusharmohan@gmail.com
 * 
 * Description: This test case reads from standard linux /etc/group
 *              and writes the output to  stdout.
 *              Statistics are printed at the end of the run.,
 */
#include <papi.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"
 
#define NUM_EVENTS 1
 
int main(int argc, char** argv) {
  int EventSet = PAPI_NULL;
  const char* names[NUM_EVENTS] = {"SELECT_USEC"};
  long long values[NUM_EVENTS];

  /* Set TESTS_QUIET variable */
  tests_quiet( argc, argv );

  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }

  /* Create the Event Set */
  if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
    fprintf(stderr, "Error creating event set\n");
    exit(2);
  }

  if (!TESTS_QUIET) printf("This program will read from stdin and echo it to stdout\n");
  int retval;
  int e;
  int event_code;
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

  int bytes = 0;
  char buf[1024];

 
//if (PAPI_read(EventSet, values) != PAPI_OK)
//   handle_error(1);
//printf("After reading the counters: %lld\n",values[0]);

  int fdready;
  fd_set readfds;
  FD_SET(0,&readfds);
 
  while (select(1,&readfds,NULL,NULL,NULL)) {
    bytes = read(0, buf, 1024);
    if (bytes > 0) write(1, buf, bytes);
    if (bytes == 0) break;
  }


  /* Stop counting events */
  if (PAPI_stop(EventSet, values) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop\n");
  }
 
  if (!TESTS_QUIET) { 
    printf("----\n");
    for (e=0; e<NUM_EVENTS; e++)  
      printf("%s: %lld\n", names[e], values[e]);
  }
  test_pass( __FILE__ );
  return 0;
}
