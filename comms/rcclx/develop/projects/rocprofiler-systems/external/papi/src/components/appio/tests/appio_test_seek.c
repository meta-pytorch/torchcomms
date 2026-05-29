/* 
 * Test case for appio
 * Author: Tushar Mohan
 *         tusharmohan@gmail.com
 * 
 * Description: This test case does a strided read of /etc/group
 *              and writes the output to  stdout.
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
 
#define NUM_EVENTS 7
 
int main(int argc, char** argv) {
  int EventSet = PAPI_NULL; 
  const char* names[NUM_EVENTS] = {"READ_CALLS", "READ_BYTES", "READ_BLOCK_SIZE", "READ_USEC", "SEEK_CALLS", "SEEK_USEC", "SEEK_ABS_STRIDE_SIZE"};
  long long values[NUM_EVENTS];

  char *infile = "/etc/group";

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

  int fdin;
  if (!TESTS_QUIET) printf("This program will do a strided read %s and write it to stdout\n", infile);
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

  fdin=open(infile, O_RDONLY);
  if (fdin < 0) perror("Could not open file for reading: \n");
  int bytes = 0;
  char buf[1024];

 
//if (PAPI_read(EventSet, values) != PAPI_OK)
//   handle_error(1);
//printf("After reading the counters: %lld\n",values[0]);

  while ((bytes = read(fdin, buf, 32)) > 0) {
    write(1, buf, bytes);
    lseek(fdin, 16, SEEK_CUR);
  }

  /* Closing the descriptors before doing the PAPI_stop
     means, OPEN_FDS will be reported as zero, which is
     right, since at the time of PAPI_stop, the descriptors
     we opened have been closed */
  close (fdin);

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
