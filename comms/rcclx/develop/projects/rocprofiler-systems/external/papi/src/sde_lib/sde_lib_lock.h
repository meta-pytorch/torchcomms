#if !defined(PAPI_SDE_LIB_LOCK_H)
#define PAPI_SDE_LIB_LOCK_H

#include "atomic_ops.h"
#if defined(AO_HAVE_test_and_set_acquire)
#define USE_LIBAO_ATOMICS
#endif

/*************************************************************************/
/* Locking functions similar to the PAPI locking function.               */
/*************************************************************************/
#if defined(USE_LIBAO_ATOMICS)

extern AO_TS_t _sde_hwd_lock_data;
#define sde_lock() {while (AO_test_and_set_acquire(&_sde_hwd_lock_data) != AO_TS_CLEAR) { ; } }
#define sde_unlock() { AO_CLEAR(&_sde_hwd_lock_data); }

#else //defined(USE_LIBAO_ATOMICS)

#include <pthread.h>

extern pthread_mutex_t _sde_hwd_lock_data;

#define  sde_lock()                          \
do{                                          \
  pthread_mutex_lock(&_sde_hwd_lock_data);   \
} while(0)

#define  sde_unlock(lck)                     \
do{                                          \
  pthread_mutex_unlock(&_sde_hwd_lock_data); \
} while(0)

#endif //defined(USE_LIBAO_ATOMICS)



#endif //!define(PAPI_SDE_LIB_LOCK_H)
