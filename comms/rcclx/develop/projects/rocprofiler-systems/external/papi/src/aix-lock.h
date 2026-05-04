#include <sys/atomic_op.h>

/* Locks */
extern atomic_p lock[];

#define _papi_hwd_lock(lck)                       \
{                                                 \
  while(_check_lock(lock[lck],0,1) == TRUE) { ; } \
}

#define _papi_hwd_unlock(lck)                   \
{                                               \
  _clear_lock(lock[lck], 0);                    \
}

