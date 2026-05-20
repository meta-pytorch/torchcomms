#ifndef _LINUX_CONTEXT_H
#define _LINUX_CONTEXT_H

/* Signal handling functions */

#undef hwd_siginfo_t

/* Changed from struct siginfo due to POSIX and Fedora 18       */
/* If this breaks anything then we need to add an aufoconf test */
typedef siginfo_t hwd_siginfo_t;

#undef hwd_ucontext_t
typedef ucontext_t hwd_ucontext_t;

#if defined(__ia64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.sc_ip
#elif defined(__i386__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_EIP]
#elif defined(__NEC__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_RIP]
#elif defined(__x86_64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_RIP]
#elif defined(__powerpc__) && !defined(__powerpc64__)
/*
 * The index of the Next IP (REG_NIP) was obtained by looking at kernel
 * source code.  It wasn't documented anywhere else that I could find.
 */
#define REG_NIP 32
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.uc_regs->gregs[REG_NIP]
#elif defined(__powerpc64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.regs->nip
#elif defined(__sparc__)
#define OVERFLOW_ADDRESS(ctx) ((struct sigcontext *)ctx.ucontext)->si_regs.pc
#elif defined(__arm__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.arm_pc
#elif defined(__aarch64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.pc
#elif defined(__mips__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.pc
#elif defined(__hppa__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.sc_iaoq[0]
#elif defined(__riscv)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.__gregs[REG_PC]
#else
#error "OVERFLOW_ADDRESS() undefined!"
#endif

#define GET_OVERFLOW_ADDRESS(ctx) (vptr_t)(OVERFLOW_ADDRESS(ctx))

#endif
