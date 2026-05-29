#ifndef _PAPI_SOLARIS_H
#define _PAPI_SOLARIS_H

#include <synch.h>
#include <procfs.h>
#include <libcpc.h>
#include <sys/procset.h>
#include <syms.h>

int _solaris_update_shlib_info( papi_mdi_t *mdi );
int _solaris_get_system_info( papi_mdi_t *mdi );
long long _solaris_get_real_usec( void );
long long _solaris_get_real_cycles( void );
long long _solaris_get_virt_usec( void );

/* Assembler prototypes */

extern void cpu_sync( void );
extern vptr_t _start, _end, _etext, _edata;

extern rwlock_t lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);

#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);

#endif

#if 0

#include <sys/asm_linkage.h>
	! #include "solaris-ultra.h"

	! These functions blatantly stolen from perfmon
	! The author of the package "perfmon" is Richard J. Enbody
	! and the home page for "perfmon" is
	! http://www.cps.msu.edu/~enbody/perfmon/index.html

	!
	! extern void cpu_sync(void);
	!
	! Make sure all instructinos and memory references before us
	! have been completed.
	.global cpu_sync
	ENTRY(cpu_sync)
	membar	#Sync		! Wait for all outstanding things to finish
	retl			! Return to the caller
	  nop			! Delay slot
	SET_SIZE(cpu_sync)

	!
	! extern unsigned long long get_tick(void)
	!
	! Read the tick register and return it
	.global get_tick
	ENTRY(get_tick)
	rd	%tick, %o0	! Get the current value of TICK
	clruw   %o0, %o1	! put the lower 32 bits into %o1
	retl			! Return to the caller
	  srlx  %o0, 32, %o0    ! put the upper 32 bits into %o0
	SET_SIZE(get_tick)

#endif





