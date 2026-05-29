#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include "papi.h"

#define NUM_EVENTS 6
static int EventSet = PAPI_NULL;
static const char* names[NUM_EVENTS] = {"READ_CALLS", "READ_BYTES","READ_USEC","WRITE_CALLS","WRITE_BYTES","WRITE_USEC"};
static long long values[NUM_EVENTS];

__attribute__ ((constructor)) void my_init(void) {
  //fprintf(stderr, "appio: constructor started\n");
  int version = PAPI_library_init (PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init version mismatch\n");
    exit(1);
  }
  else {
    fprintf(stderr, "appio: PAPI library initialized\n");
  }

  /* Create the Event Set */
  if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
    fprintf(stderr, "Error creating event set\n");
    exit(2);
  }

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
  fprintf(stderr, "appio: starting PAPI counters; main program will follow\n");
  if (PAPI_start(EventSet) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_start\n");
    exit(1);
  }
  return;
}

__attribute__ ((destructor)) void my_fini(void) {
  int e;
  //fprintf(stderr, "appio: destructor called\n");
  if (PAPI_stop(EventSet, values) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_stop_counters\n");
  }
  fprintf(stderr, "\nappio: PAPI counts (for pid=%6d)\n"
                  "appio: ----------------------------\n", (int)getpid());
  for (e=0; e<NUM_EVENTS; e++)
    fprintf(stderr, "appio: %s : %lld\n", names[e], values[e]);
  return;
}
