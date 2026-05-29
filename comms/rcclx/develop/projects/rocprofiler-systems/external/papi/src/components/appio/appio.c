/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    appio.c
 *
 * @author  Philip Mucci
 *          phil.mucci@samaratechnologygroup.com
 *
 * @author  Tushar Mohan
 *          tusharmohan@gmail.com
 *
 * Credit to: 
 *          Jose Pedro Oliveira
 *          jpo@di.uminho.pt
 * whose code in the linux net component was used as a template for
 * many sections of code in this component.
 *
 * @ingroup papi_components
 *
 * @brief appio component
 *  This file contains the source code for a component that enables
 *  PAPI to access application level file and socket I/O information.
 *  It does this through function replacement in the first person and
 *  by trapping syscalls in the third person.
 */

#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "appio.h"

// The PIC test implies it's built for shared linkage
#ifdef PIC
#  include "dlfcn.h"
#endif

/*
#pragma weak dlerror
static void *_dlsym_fake(void *handle, const char* symbol) { (void) handle; (void) symbol; return NULL; }
void *dlsym(void *handle, const char* symbol) __attribute__ ((weak, alias ("_dlsym_fake")));
*/

papi_vector_t _appio_vector;

/*********************************************************************
 * Private
 ********************************************************************/

//#define APPIO_FOO 1

static APPIO_native_event_entry_t * _appio_native_events;


/* If you modify the appio_stats_t below, you MUST update APPIO_MAX_COUNTERS */
static __thread long long _appio_register_current[APPIO_MAX_COUNTERS];
typedef enum {
  READ_BYTES = 0,
  READ_CALLS,
  READ_ERR,
  READ_INTERRUPTED,
  READ_WOULD_BLOCK,
  READ_SHORT,
  READ_EOF,
  READ_BLOCK_SIZE,
  READ_USEC,
  WRITE_BYTES,
  WRITE_CALLS,
  WRITE_ERR,
  WRITE_SHORT,
  WRITE_INTERRUPTED,
  WRITE_WOULD_BLOCK,
  WRITE_BLOCK_SIZE,
  WRITE_USEC,
  OPEN_CALLS,
  OPEN_ERR,
  OPEN_FDS,
  SELECT_USEC,
  RECV_BYTES,
  RECV_CALLS,
  RECV_ERR,
  RECV_INTERRUPTED,
  RECV_WOULD_BLOCK,
  RECV_SHORT,
  RECV_EOF,
  RECV_BLOCK_SIZE,
  RECV_USEC,
  SOCK_READ_BYTES,
  SOCK_READ_CALLS,
  SOCK_READ_ERR,
  SOCK_READ_SHORT,
  SOCK_READ_WOULD_BLOCK,
  SOCK_READ_USEC,
  SOCK_WRITE_BYTES,
  SOCK_WRITE_CALLS,
  SOCK_WRITE_ERR,
  SOCK_WRITE_SHORT,
  SOCK_WRITE_WOULD_BLOCK,
  SOCK_WRITE_USEC,
  SEEK_CALLS,
  SEEK_ABS_STRIDE_SIZE,
  SEEK_USEC
} _appio_stats_t ;

static const struct appio_counters {
    const char *name;
    const char *description;
} _appio_counter_info[APPIO_MAX_COUNTERS] = {
    { "READ_BYTES",      "Bytes read"},
    { "READ_CALLS",      "Number of read calls"},
    { "READ_ERR",        "Number of read calls that resulted in an error"},
    { "READ_INTERRUPTED","Number of read calls that timed out or were interruped"},
    { "READ_WOULD_BLOCK","Number of read calls that would have blocked"},
    { "READ_SHORT",      "Number of read calls that returned less bytes than requested"},
    { "READ_EOF",        "Number of read calls that returned an EOF"},
    { "READ_BLOCK_SIZE", "Average block size of reads"},
    { "READ_USEC",       "Real microseconds spent in reads"},
    { "WRITE_BYTES",     "Bytes written"},
    { "WRITE_CALLS",     "Number of write calls"},
    { "WRITE_ERR",       "Number of write calls that resulted in an error"},
    { "WRITE_SHORT",     "Number of write calls that wrote less bytes than requested"},
    { "WRITE_INTERRUPTED","Number of write calls that timed out or were interrupted"},
    { "WRITE_WOULD_BLOCK","Number of write calls that would have blocked"},
    { "WRITE_BLOCK_SIZE","Mean block size of writes"},
    { "WRITE_USEC",      "Real microseconds spent in writes"},
    { "OPEN_CALLS",      "Number of open calls"},
    { "OPEN_ERR",        "Number of open calls that resulted in an error"},
    { "OPEN_FDS",        "Number of currently open descriptors"},
    { "SELECT_USEC",     "Real microseconds spent in select calls"},
    { "RECV_BYTES",      "Bytes read in recv/recvmsg/recvfrom"},
    { "RECV_CALLS",      "Number of recv/recvmsg/recvfrom calls"},
    { "RECV_ERR",        "Number of recv/recvmsg/recvfrom calls that resulted in an error"},
    { "RECV_INTERRUPTED","Number of recv/recvmsg/recvfrom calls that timed out or were interruped"},
    { "RECV_WOULD_BLOCK","Number of recv/recvmsg/recvfrom calls that would have blocked"},
    { "RECV_SHORT",      "Number of recv/recvmsg/recvfrom calls that returned less bytes than requested"},
    { "RECV_EOF",        "Number of recv/recvmsg/recvfrom calls that returned an EOF"},
    { "RECV_BLOCK_SIZE", "Average block size of recv/recvmsg/recvfrom"},
    { "RECV_USEC",       "Real microseconds spent in recv/recvmsg/recvfrom"},
    { "SOCK_READ_BYTES", "Bytes read from socket"},
    { "SOCK_READ_CALLS", "Number of read calls on socket"},
    { "SOCK_READ_ERR",   "Number of read calls on socket that resulted in an error"},
    { "SOCK_READ_SHORT", "Number of read calls on socket that returned less bytes than requested"},
    { "SOCK_READ_WOULD_BLOCK", "Number of read calls on socket that would have blocked"},
    { "SOCK_READ_USEC",  "Real microseconds spent in read(s) on socket(s)"},
    { "SOCK_WRITE_BYTES","Bytes written to socket"},
    { "SOCK_WRITE_CALLS","Number of write calls to socket"},
    { "SOCK_WRITE_ERR",  "Number of write calls to socket that resulted in an error"},
    { "SOCK_WRITE_SHORT","Number of write calls to socket that wrote less bytes than requested"},
    { "SOCK_WRITE_WOULD_BLOCK","Number of write calls to socket that would have blocked"},
    { "SOCK_WRITE_USEC", "Real microseconds spent in write(s) to socket(s)"},
    { "SEEK_CALLS",      "Number of seek calls"},
    { "SEEK_ABS_STRIDE_SIZE", "Average absolute stride size of seeks"},
    { "SEEK_USEC",       "Real microseconds spent in seek calls"}
};

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

/*********************************************************************
 ***  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ****
 ********************************************************************/

int __close(int fd);
int close(int fd) {
  int retval;
  SUBDBG("appio: intercepted close(%d)\n", fd);
  retval = __close(fd);
  if ((retval == 0) && (_appio_register_current[OPEN_FDS]>0)) _appio_register_current[OPEN_FDS]--;
  return retval;
}

int __open(const char *pathname, int flags, mode_t mode);
int open(const char *pathname, int flags, mode_t mode) {
  int retval;
  SUBDBG("appio: intercepted open(%s,%d,%d)\n", pathname, flags, mode);
  retval = __open(pathname,flags,mode);
  _appio_register_current[OPEN_CALLS]++;
  if (retval < 0) _appio_register_current[OPEN_ERR]++;
  else _appio_register_current[OPEN_FDS]++;
  return retval;
}

/* we use timeval as a zero value timeout to select in read/write
   for polling if the operation would block */
struct timeval zerotv; /* this has to be zero, so define it here */

int __select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout) {
  int retval;
  SUBDBG("appio: intercepted select(%d,%p,%p,%p,%p)\n", nfds,readfds,writefds,exceptfds,timeout);
  long long start_ts = PAPI_get_real_usec();
  retval = __select(nfds,readfds,writefds,exceptfds,timeout);
  long long duration = PAPI_get_real_usec() - start_ts;
  _appio_register_current[SELECT_USEC] += duration;
  return retval;
}

off_t __lseek(int fd, off_t offset, int whence);
off_t lseek(int fd, off_t offset, int whence) {
  off_t retval;
  SUBDBG("appio: intercepted lseek(%d,%ld,%d)\n", fd, offset, whence);
  long long start_ts = PAPI_get_real_usec();
  retval = __lseek(fd, offset, whence);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[SEEK_CALLS]++;
  _appio_register_current[SEEK_USEC] += duration;
  if (offset < 0) offset = -offset; // get abs offset
  _appio_register_current[SEEK_ABS_STRIDE_SIZE]= (n * _appio_register_current[SEEK_ABS_STRIDE_SIZE] + offset)/(n+1); // mean absolute stride size
  return retval;
}

extern int errno;
ssize_t __read(int fd, void *buf, size_t count);
ssize_t read(int fd, void *buf, size_t count) {
  int retval;
  SUBDBG("appio: intercepted read(%d,%p,%lu)\n", fd, buf, (unsigned long)count);

  struct stat st;
  int issocket = 0;
  if (fstat(fd, &st) == 0) {
    if ((st.st_mode & S_IFMT) == S_IFSOCK) issocket = 1;
  }
  // check if read would block on descriptor
  fd_set readfds;
  FD_ZERO(&readfds);
  FD_SET(fd, &readfds);
  int ready = __select(fd+1, &readfds, NULL, NULL, &zerotv);
  if (ready == 0) {
     _appio_register_current[READ_WOULD_BLOCK]++; 
     if (issocket) _appio_register_current[SOCK_READ_WOULD_BLOCK]++; 
  }

  long long start_ts = PAPI_get_real_usec();
  retval = __read(fd,buf, count);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[READ_CALLS]++; // read calls
  if (issocket) _appio_register_current[SOCK_READ_CALLS]++; // read calls
  if (retval > 0) {
    _appio_register_current[READ_BLOCK_SIZE]= (n * _appio_register_current[READ_BLOCK_SIZE] + count)/(n+1); // mean size
    _appio_register_current[READ_BYTES] += retval; // read bytes
    if (issocket) _appio_register_current[SOCK_READ_BYTES] += retval;
    if (retval < (int)count) {
       _appio_register_current[READ_SHORT]++; // read short
       if (issocket) _appio_register_current[SOCK_READ_SHORT]++; // read short
    }
    _appio_register_current[READ_USEC] += duration;
    if (issocket) _appio_register_current[SOCK_READ_USEC] += duration;
  }
  if (retval < 0) { 
    _appio_register_current[READ_ERR]++; // read err
    if (issocket) _appio_register_current[SOCK_READ_ERR]++; // read err
    if (EINTR == errno)
      _appio_register_current[READ_INTERRUPTED]++; // signal interrupted the read
    //if ((EAGAIN == errno) || (EWOULDBLOCK == errno)) {
    //  _appio_register_current[READ_WOULD_BLOCK]++; //read would block on descriptor marked as non-blocking
    //  if (issocket) _appio_register_current[SOCK_READ_WOULD_BLOCK]++; //read would block on descriptor marked as non-blocking
    //}
  }
  if (retval == 0) _appio_register_current[READ_EOF]++; // read eof
  return retval;
}

size_t _IO_fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t retval;
  SUBDBG("appio: intercepted fread(%p,%lu,%lu,%p)\n", ptr, (unsigned long) size, (unsigned long) nmemb, (void*) stream);
  long long start_ts = PAPI_get_real_usec();
  retval = _IO_fread(ptr,size,nmemb,stream);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[READ_CALLS]++; // read calls
  if (retval > 0) {
    _appio_register_current[READ_BLOCK_SIZE]= (n * _appio_register_current[READ_BLOCK_SIZE]+ size*nmemb)/(n+1);//mean size
    _appio_register_current[READ_BYTES]+= retval * size; // read bytes
    if (retval < nmemb) _appio_register_current[READ_SHORT]++; // read short
    _appio_register_current[READ_USEC] += duration;
  }

  /* A value of zero returned means one of two things..*/
  if (retval == 0) {
     if (feof(stream)) _appio_register_current[READ_EOF]++; // read eof
     else _appio_register_current[READ_ERR]++; // read err
  }
  return retval;
}

ssize_t __write(int fd, const void *buf, size_t count);
ssize_t write(int fd, const void *buf, size_t count) {
  int retval;
  SUBDBG("appio: intercepted write(%d,%p,%lu)\n", fd, buf, (unsigned long)count);
  struct stat st;
  int issocket = 0;
  if (fstat(fd, &st) == 0) {
    if ((st.st_mode & S_IFMT) == S_IFSOCK) issocket = 1;
  }

  // check if write would block on descriptor
  fd_set writefds;
  FD_ZERO(&writefds);
  FD_SET(fd, &writefds);
  int ready = __select(fd+1, NULL, &writefds, NULL, &zerotv);
  if (ready == 0) {
    _appio_register_current[WRITE_WOULD_BLOCK]++; 
    if (issocket) _appio_register_current[SOCK_WRITE_WOULD_BLOCK]++; 
  }

  long long start_ts = PAPI_get_real_usec();
  retval = __write(fd,buf, count);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[WRITE_CALLS]++; // write calls
  if (issocket) _appio_register_current[SOCK_WRITE_CALLS]++; // socket write
  if (retval >= 0) {
    _appio_register_current[WRITE_BLOCK_SIZE]= (n * _appio_register_current[WRITE_BLOCK_SIZE] + count)/(n+1); // mean size
    _appio_register_current[WRITE_BYTES]+= retval; // write bytes
    if (issocket) _appio_register_current[SOCK_WRITE_BYTES] += retval;
    if (retval < (int)count) {
      _appio_register_current[WRITE_SHORT]++; // short write
      if (issocket) _appio_register_current[SOCK_WRITE_SHORT]++; 
    }
    _appio_register_current[WRITE_USEC] += duration;
    if (issocket) _appio_register_current[SOCK_WRITE_USEC] += duration;
  }
  if (retval < 0) {
    _appio_register_current[WRITE_ERR]++; // err
    if (issocket) _appio_register_current[SOCK_WRITE_ERR]++;
    if (EINTR == errno)
      _appio_register_current[WRITE_INTERRUPTED]++; // signal interrupted the op
    //if ((EAGAIN == errno) || (EWOULDBLOCK == errno)) {
    //  _appio_register_current[WRITE_WOULD_BLOCK]++; //op would block on descriptor marked as non-blocking
    //  if (issocket) _appio_register_current[SOCK_WRITE_WOULD_BLOCK]++;
    //}
  }
  return retval;
}

// The PIC test implies it's built for shared linkage
#ifdef PIC
static ssize_t (*__recv)(int sockfd, void *buf, size_t len, int flags) = NULL;
ssize_t recv(int sockfd, void *buf, size_t len, int flags) {
  int retval;
  SUBDBG("appio: intercepted recv(%d,%p,%lu,%d)\n", sockfd, buf, (unsigned long)len, flags);
  if (!__recv) __recv  = dlsym(RTLD_NEXT, "recv");
  if (!__recv) {
    fprintf(stderr, "appio,c Internal Error: Could not obtain handle for real recv\n");
    exit(1);
  }
  // check if recv would block on descriptor
  fd_set readfds;
  FD_ZERO(&readfds);
  FD_SET(sockfd, &readfds);
  int ready = __select(sockfd+1, &readfds, NULL, NULL, &zerotv);
  if (ready == 0) _appio_register_current[RECV_WOULD_BLOCK]++; 

  long long start_ts = PAPI_get_real_usec();
  retval = __recv(sockfd, buf, len, flags);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[RECV_CALLS]++; // read calls
  if (retval > 0) {
    _appio_register_current[RECV_BLOCK_SIZE]= (n * _appio_register_current[RECV_BLOCK_SIZE] + len)/(n+1); // mean size
    _appio_register_current[RECV_BYTES] += retval; // read bytes
    if (retval < (int)len) _appio_register_current[RECV_SHORT]++; // read short
    _appio_register_current[RECV_USEC] += duration;
  }
  if (retval < 0) { 
    _appio_register_current[RECV_ERR]++; // read err
    if (EINTR == errno)
      _appio_register_current[RECV_INTERRUPTED]++; // signal interrupted the read
    if ((EAGAIN == errno) || (EWOULDBLOCK == errno)) 
      _appio_register_current[RECV_WOULD_BLOCK]++; //read would block on descriptor marked as non-blocking
  }
  if (retval == 0) _appio_register_current[RECV_EOF]++; // read eof
  return retval;
}
#endif /* PIC */

size_t _IO_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t retval;
  SUBDBG("appio: intercepted fwrite(%p,%lu,%lu,%p)\n", ptr, (unsigned long) size, (unsigned long) nmemb, (void*) stream);
  long long start_ts = PAPI_get_real_usec();
  retval = _IO_fwrite(ptr,size,nmemb,stream);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[WRITE_CALLS]++; // write calls
  if (retval > 0) {
    _appio_register_current[WRITE_BLOCK_SIZE]= (n * _appio_register_current[WRITE_BLOCK_SIZE] + size*nmemb)/(n+1); // mean block size
    _appio_register_current[WRITE_BYTES]+= retval * size; // write bytes
    if (retval < nmemb) _appio_register_current[WRITE_SHORT]++; // short write
    _appio_register_current[WRITE_USEC] += duration;
  }
  if (retval == 0) _appio_register_current[WRITE_ERR]++; // err
  return retval;
}


/*********************************************************************
 ***************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *********
 *********************************************************************/

/*
 * This is called whenever a thread is initialized
 */
static int
_appio_init_thread( hwd_context_t *ctx )
{
    ( void ) ctx;
    SUBDBG("_appio_init_thread %p\n", ctx);
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */

static int
_appio_init_component( int cidx  )
{
    int strErr;
    int retval = PAPI_OK;
    SUBDBG("_appio_component %d\n", cidx);
    _appio_native_events = (APPIO_native_event_entry_t *) papi_calloc(APPIO_MAX_COUNTERS, sizeof(APPIO_native_event_entry_t));

    if (_appio_native_events == NULL ) {
      PAPIERROR( "malloc():Could not get memory for events table" );
      strErr=snprintf(_appio_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "malloc() failed in %s for %lu bytes.", __func__, APPIO_MAX_COUNTERS*sizeof(APPIO_native_event_entry_t));
      _appio_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
      if (strErr>PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
      retval = PAPI_ENOMEM;
      goto fn_fail;
    }
    int i;
    for (i=0; i<APPIO_MAX_COUNTERS; i++) {
      _appio_native_events[i].name = _appio_counter_info[i].name;
      _appio_native_events[i].description = _appio_counter_info[i].description;
      _appio_native_events[i].resources.selector = i + 1;
    }
  
    /* Export the total number of events available */
    _appio_vector.cmp_info.num_native_events = APPIO_MAX_COUNTERS;;

    /* Export the component id */
    _appio_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int
_appio_init_control_state( hwd_control_state_t *ctl )
{
    ( void ) ctl;

    return PAPI_OK;
}


static int
_appio_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;

    SUBDBG("_appio_start %p %p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;

    /* this memset needs to move to thread_init */
    memset(_appio_register_current, 0, APPIO_MAX_COUNTERS * sizeof(_appio_register_current[0]));

    /* set initial values to 0 */
    memset(appio_ctl->values, 0, APPIO_MAX_COUNTERS*sizeof(appio_ctl->values[0]));
    
    return PAPI_OK;
}


static int
_appio_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
    long long ** events, int flags )
{
    (void) flags;
    (void) ctx;

    SUBDBG("_appio_read %p %p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    int i;

    for ( i=0; i<appio_ctl->num_events; i++ ) {
            int index = appio_ctl->counter_bits[i];
            SUBDBG("event=%d, index=%d, val=%lld\n", i, index, _appio_register_current[index]);
            appio_ctl->values[index] = _appio_register_current[index];
    }
    *events = appio_ctl->values;

    return PAPI_OK;
}


static int
_appio_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;

    SUBDBG("_appio_stop ctx=%p ctl=%p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    int i;
    for ( i=0; i<appio_ctl->num_events; i++ ) {
            int index = appio_ctl->counter_bits[i];
            SUBDBG("event=%d, index=%d, val=%lld\n", i, index, _appio_register_current[index]);
            appio_ctl->values[i] = _appio_register_current[index];
    }

    return PAPI_OK;
}


/*
 * Thread shutdown
 */
static int
_appio_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}


/*
 * Clean up what was setup in appio_init_component().
 */
static int
_appio_shutdown_component( void )
{
    papi_free( _appio_native_events );
    return PAPI_OK;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and
 * PAPI_SET_INHERIT
 */
static int
_appio_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


static int
_appio_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t *native, int count, hwd_context_t *ctx )
{
    ( void ) ctx;
    ( void ) ctl;

    SUBDBG("_appio_update_control_state ctx=%p ctl=%p num_events=%d\n", ctx, ctl, count);
    int i, index;
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    (void) ctx;

    for ( i = 0; i < count; i++ ) {
        index = native[i].ni_event;
        appio_ctl->counter_bits[i] = index;
        native[i].ni_position = index;
    }
    appio_ctl->num_events = count;

    return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER   is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL    is all of the domains
 */
static int
_appio_set_domain( hwd_control_state_t *ctl, int domain )
{
    ( void ) ctl;

    int found = 0;

    if ( PAPI_DOM_USER == domain )   found = 1;

    if ( !found )
        return PAPI_EINVAL;

    return PAPI_OK;
}


static int
_appio_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;

    return PAPI_OK;
}


/*
 * Native Event functions
 */
static int
_appio_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;

    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            *EventCode = 0;
            return PAPI_OK;
            break;

        case PAPI_ENUM_EVENTS:
            index = *EventCode;
            if ( index < APPIO_MAX_COUNTERS - 1 ) {
                *EventCode = *EventCode + 1;
                return PAPI_OK;
            } else {
                return PAPI_ENOEVNT;
            }
            break;

        default:
            return PAPI_EINVAL;
            break;
    }
    return PAPI_EINVAL;
}


/*
 *
 */
static int
_appio_ntv_name_to_code( const char *name, unsigned int *EventCode )
{
    int i;

    for ( i=0; i<APPIO_MAX_COUNTERS; i++) {
        if (strcmp(name, _appio_counter_info[i].name) == 0) {
            *EventCode = i;
            return PAPI_OK;
        }
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_appio_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        strncpy( name, _appio_counter_info[index].name, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_appio_ntv_code_to_descr( unsigned int EventCode, char *desc, int len )
{
    int index = EventCode;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        strncpy(desc, _appio_counter_info[index].description, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_appio_ntv_code_to_bits( unsigned int EventCode, hwd_register_t *bits )
{
    int index = EventCode;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        memcpy( ( APPIO_register_t * ) bits,
                &( _appio_native_events[index].resources ),
                sizeof ( APPIO_register_t ) );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
papi_vector_t _appio_vector = {
    .cmp_info = {
        /* default component information (unspecified values are initialized to 0) */
        .name                  = "appio",
        .short_name            = "appio",
        .version               = "1.1.2.4",
        .CmpIdx                = 0,              /* set by init_component */
        .num_mpx_cntrs         = APPIO_MAX_COUNTERS,
        .num_cntrs             = APPIO_MAX_COUNTERS,
        .default_domain        = PAPI_DOM_USER,
        .available_domains   = PAPI_DOM_USER,
        .default_granularity   = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig     = PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
        .fast_real_timer       = 0,
        .fast_virtual_timer    = 0,
        .attach                = 0,
        .attach_must_ptrace    = 0,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context               = sizeof ( APPIO_context_t ),
        .control_state         = sizeof ( APPIO_control_state_t ),
        .reg_value             = sizeof ( APPIO_register_t ),
        .reg_alloc             = sizeof ( APPIO_reg_alloc_t ),
    },

    /* function pointers in this component */
    .init_thread               = _appio_init_thread,
    .init_component            = _appio_init_component,
    .init_control_state        = _appio_init_control_state,
    .start                     = _appio_start,
    .stop                      = _appio_stop,
    .read                      = _appio_read,
    .shutdown_thread           = _appio_shutdown_thread,
    .shutdown_component        = _appio_shutdown_component,
    .ctl                       = _appio_ctl,

    .update_control_state      = _appio_update_control_state,
    .set_domain                = _appio_set_domain,
    .reset                     = _appio_reset,

    .ntv_enum_events           = _appio_ntv_enum_events,
    .ntv_name_to_code          = _appio_ntv_name_to_code,
    .ntv_code_to_name          = _appio_ntv_code_to_name,
    .ntv_code_to_descr         = _appio_ntv_code_to_descr,
    .ntv_code_to_bits          = _appio_ntv_code_to_bits
    /* .ntv_bits_to_info          = NULL, */
};

/* vim:set ts=4 sw=4 sts=4 et: */
