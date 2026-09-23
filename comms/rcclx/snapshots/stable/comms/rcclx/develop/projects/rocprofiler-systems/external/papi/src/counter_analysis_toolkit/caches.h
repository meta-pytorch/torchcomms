#ifndef _CACHES_
#define _CACHES_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
// Header files for uintptr_t
#if defined (__SVR4) && defined (__sun)
# include <sys/types.h>
#else
# include <stdint.h>
#endif
#include <unistd.h>

// Header files for setting the affinity
#if defined(__linux__) 
#  define __USE_GNU 1
#  include <sched.h>
#elif defined (__SVR4) && defined (__sun)
//#elif defined(__sparc)
#  include <sys/types.h>
#  include <sys/processor.h>
#  include <sys/procset.h>
#endif

#include <pthread.h>

#define SIZE (512*1024)

#define L_SIZE 0
#define C_SIZE 1
#define ASSOC  2

//#define DEBUG

#define MAXTHREADS 128
typedef struct run_output_s{
    double dt[MAXTHREADS];
    double counter[MAXTHREADS];
    int status;
}run_output_t;

static inline double getticks(void){
     double ret;
     struct timeval tv;

     gettimeofday(&tv, NULL);
     ret = 1000*1000*(double)tv.tv_sec + (double)tv.tv_usec;
     return ret;
}

static inline double elapsed(double t1, double t0){ 
     return (double)t1 - (double)t0;
} 

extern int compar_lf(const void *a, const void *b);
extern int compar_lld(const void *a, const void *b);

#endif
