#ifndef _LINUX_COMMON_H
#define _LINUX_COMMON_H

#define LINUX_VERSION(a,b,c) ( ((a&0xff)<<24) | ((b&0xff)<<16) | ((c&0xff) << 8))

#define min(x, y) ({				\
	typeof(x) _min1 = (x);			\
	typeof(y) _min2 = (y);			\
	(void) (&_min1 == &_min2);		\
	_min1 < _min2 ? _min1 : _min2; })

static inline pid_t
mygettid( void )
{
#if defined(__NEC__)
    return 0;
#else
#ifdef SYS_gettid
	return syscall( SYS_gettid );
#elif defined(__NR_gettid)
	return syscall( __NR_gettid );
#else
#error "cannot find gettid"
#endif
#endif
}

#ifndef F_SETOWN_EX
   #define F_SETOWN_EX     15
   #define F_GETOWN_EX     16
   
   #define F_OWNER_TID     0
   #define F_OWNER_PID     1
   #define F_OWNER_PGRP    2
   
   struct f_owner_ex {
              int     type;
              pid_t   pid;
   };
#endif

int _linux_detect_nmi_watchdog();

#if HAVE_SCHED_GETCPU
#include <sched.h>
/* If possible, pick the processors the code is currently running on. */
#define _papi_getcpu()   sched_getcpu()
#else
/* Just map to processor 0 if sched_getcpu() is not available. */
#define _papi_getcpu()   0

#endif

#endif
