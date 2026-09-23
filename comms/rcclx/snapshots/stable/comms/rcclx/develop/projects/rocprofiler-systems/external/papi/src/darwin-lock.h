#ifndef _DARWIN_LOCK_H
#define _DARWIN_LOCK_H

#include "mb.h"

/* Locking functions */

#if defined(USE_PTHREAD_MUTEXES)

#include <pthread.h>

extern pthread_mutex_t _papi_hwd_lock_data[PAPI_MAX_LOCK];

#define  _papi_hwd_lock(lck)                       \
do                                                 \
{                                                  \
   pthread_mutex_lock (&_papi_hwd_lock_data[lck]); \
} while(0)
#define  _papi_hwd_unlock(lck)                     \
do                                                 \
{                                                  \
  pthread_mutex_unlock(&_papi_hwd_lock_data[lck]); \
} while(0)


#else

extern volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#define MUTEX_OPEN 0
#define MUTEX_CLOSED 1

#define  _papi_hwd_lock(lck)                    \
do                                              \
{                                               \
   unsigned int res = 0;                        \
   do {                                         \
      __asm__ __volatile__ ("lock ; " "cmpxchg %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");  \
   } while(res != (unsigned int)MUTEX_OPEN);   \
} while(0)
#define  _papi_hwd_unlock(lck)                  \
do                                              \
{                                               \
   unsigned int res = 0;                       \
   __asm__ __volatile__ ("xchg %0,%1" : "=r"(res) : "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");                                \
} while(0)

#endif 
#endif
