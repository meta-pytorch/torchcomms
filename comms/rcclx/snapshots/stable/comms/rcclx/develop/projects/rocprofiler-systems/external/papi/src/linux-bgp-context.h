#ifndef _LINUX_BGP_CONTEXT_H
#define _LINUX_BGP_CONTEXT_H

#include <sys/ucontext.h>

#define GET_OVERFLOW_ADDRESS(ctx) 0x0

/* Signal handling functions */
#undef hwd_siginfo_t
#undef hwd_ucontext_t
typedef int hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#endif
