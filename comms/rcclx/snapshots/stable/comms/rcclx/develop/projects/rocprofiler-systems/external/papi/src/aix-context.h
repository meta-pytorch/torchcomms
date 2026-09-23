#ifndef _PAPI_AIX_CONTEXT_H
#define _PAPI_AIX_CONTEXT_H

/* overflow */
/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef hwd_siginfo_t
#undef hwd_ucontext_t
typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void *)(((hwd_ucontext_t *)(ctx->ucontext))->sc_jmpbuf.jmp_context.iar)

#endif /* _PAPI_AIX_CONTEXT */

