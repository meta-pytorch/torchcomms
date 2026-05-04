#ifndef _SOLARIS_CONTEXT_H
#define _SOLARIS_CONTEXT_H

#include <sys/ucontext.h>

typedef siginfo_t _solaris_siginfo_t;
#define hwd_siginfo_t _solaris_siginfo_t
typedef ucontext_t _solaris_ucontext_t;
#define hwd_ucontext_t _solaris_ucontext_t

#define GET_OVERFLOW_ADDRESS(ctx)  (void*)(ctx->ucontext->uc_mcontext.gregs[REG_PC])

#endif
