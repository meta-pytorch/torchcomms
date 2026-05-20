#include <stdio.h>
#include <string.h>
#include <stdlib.h> /* exit() */
#include <errno.h> /* herror() */
#include <netdb.h> /* gethostbyname() */
#include <sys/types.h> /* bind() accept() */
#include <sys/socket.h> /* bind() accept() */
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define PORT 3490
#define NUM_EVENTS 15

main(int argc, char *argv[]) {
  int EventSet = PAPI_NULL;
  const char* names[NUM_EVENTS] = {"READ_CALLS", "READ_BYTES", "READ_USEC", "READ_WOULD_BLOCK", "SOCK_READ_CALLS", "SOCK_READ_BYTES", "SOCK_READ_USEC", "SOCK_READ_WOULD_BLOCK", "WRITE_BYTES", "WRITE_CALLS", "WRITE_WOULD_BLOCK", "WRITE_USEC", "SOCK_WRITE_BYTES", "SOCK_WRITE_CALLS", "SOCK_WRITE_USEC"};
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

  if (!TESTS_QUIET) 
    printf("This program will listen on port 3490, and write data received to standard output AND socket\n"
           "In the output ensure that the following identities hold:\n"
           "READ_* == SOCK_READ_*\n"
           "WRITE_{CALLS,BYTES} = 2 * SOCK_WRITE_{CALLS,BYTES}\n"
           "SOCK_READ_BYTES == SOCK_WRITE_BYTES\n");
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

  int bytes = 0;
  char buf[1024];

  int sockfd, n_sockfd, sin_size, len;
  char *host_addr;
  struct sockaddr_in my_addr;
  struct sockaddr_in their_addr;
  my_addr.sin_family = AF_INET;
  my_addr.sin_port = htons(PORT);
  my_addr.sin_addr.s_addr = INADDR_ANY;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    perror("socket");
    exit(1);
  }
  if ((bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr))) == -1) {
    perror("bind");
    exit(1);
  }
  listen(sockfd, 10);
  if ((n_sockfd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size)) == -1) {
    perror("accept");
    exit(1);
  }
  close(sockfd);

  /* Start counting events */
  if (PAPI_start(EventSet) != PAPI_OK) {
    fprintf(stderr, "Error in PAPI_start\n");
    exit(1);
  }

  while ((bytes = read(n_sockfd, buf, 1024)) > 0) {
    write(1, buf, bytes);
    write(n_sockfd, buf, bytes);
  }

  close(n_sockfd);

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
