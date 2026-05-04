extern rwlock_t lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);

#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);





