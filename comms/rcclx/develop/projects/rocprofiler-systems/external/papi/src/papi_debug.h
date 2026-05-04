/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/
/** 
* @file    papi_debug.h
* @author  Philip Mucci
*          mucci@cs.utk.edu
* @author  Dan Terpstra
*          terpstra.utk.edu
* @author  Kevin London
*	       london@cs.utk.edu
* @author  Haihang You
*          you@cs.utk.edu
*/

#ifndef _PAPI_DEBUG_H
#define _PAPI_DEBUG_H

#ifdef NO_VARARG_MACRO
#include <stdarg.h>
#endif

#include <stdio.h>

/* Debug Levels */

#define DEBUG_SUBSTRATE         0x002
#define DEBUG_API               0x004
#define DEBUG_INTERNAL          0x008
#define DEBUG_THREADS           0x010
#define DEBUG_MULTIPLEX         0x020
#define DEBUG_OVERFLOW          0x040
#define DEBUG_PROFILE           0x080
#define DEBUG_MEMORY            0x100
#define DEBUG_LEAK              0x200
#define DEBUG_HIGHLEVEL         0x400
#define DEBUG_ALL               (DEBUG_SUBSTRATE|DEBUG_API|DEBUG_INTERNAL|DEBUG_THREADS|DEBUG_MULTIPLEX|DEBUG_OVERFLOW|DEBUG_PROFILE|DEBUG_MEMORY|DEBUG_LEAK|DEBUG_HIGHLEVEL)

/* Please get rid of the DBG macro from your code */

extern int _papi_hwi_debug;
extern unsigned long int ( *_papi_hwi_thread_id_fn ) ( void );

#ifdef DEBUG

#ifdef __GNUC__
#define FUNC __FUNCTION__
#elif defined(__func__)
#define FUNC __func__
#else
#define FUNC "?"
#endif

#define DEBUGLABEL(a) if (_papi_hwi_thread_id_fn) fprintf(stderr, "%s:%s:%s:%d:%d:%#lx ",a,__FILE__, FUNC, __LINE__,(int)getpid(),_papi_hwi_thread_id_fn()); else fprintf(stderr, "%s:%s:%s:%d:%d ",a,__FILE__, FUNC, __LINE__, (int)getpid())
#define ISLEVEL(a) (_papi_hwi_debug&a)

#define DEBUGLEVEL(a) ((a&DEBUG_SUBSTRATE)?"SUBSTRATE":(a&DEBUG_API)?"API":(a&DEBUG_INTERNAL)?"INTERNAL":(a&DEBUG_THREADS)?"THREADS":(a&DEBUG_MULTIPLEX)?"MULTIPLEX":(a&DEBUG_OVERFLOW)?"OVERFLOW":(a&DEBUG_PROFILE)?"PROFILE":(a&DEBUG_MEMORY)?"MEMORY":(a&DEBUG_LEAK)?"LEAK":(a&DEBUG_HIGHLEVEL)?"HIGHLEVEL":"UNKNOWN")

#ifndef NO_VARARG_MACRO		 /* Has variable arg macro support */
#define PAPIDEBUG(level,format, args...) { if(_papi_hwi_debug&level){DEBUGLABEL(DEBUGLEVEL(level));fprintf(stderr,format, ## args);}}

 /* Macros */

#define SUBDBG(format, args...) (PAPIDEBUG(DEBUG_SUBSTRATE,format, ## args))
#define APIDBG(format, args...) (PAPIDEBUG(DEBUG_API,format, ## args))
#define INTDBG(format, args...) (PAPIDEBUG(DEBUG_INTERNAL,format, ## args))
#define THRDBG(format, args...) (PAPIDEBUG(DEBUG_THREADS,format, ## args))
#define MPXDBG(format, args...) (PAPIDEBUG(DEBUG_MULTIPLEX,format, ## args))
#define OVFDBG(format, args...) (PAPIDEBUG(DEBUG_OVERFLOW,format, ## args))
#define PRFDBG(format, args...) (PAPIDEBUG(DEBUG_PROFILE,format, ## args))
#define MEMDBG(format, args...) (PAPIDEBUG(DEBUG_MEMORY,format, ## args))
#define LEAKDBG(format, args...) (PAPIDEBUG(DEBUG_LEAK,format, ## args))
#define HLDBG(format, args...) (PAPIDEBUG(DEBUG_HIGHLEVEL,format, ## args))
#endif

#else
#ifndef NO_VARARG_MACRO		 /* Has variable arg macro support */
#define SUBDBG(format, args...) { ; }
#define APIDBG(format, args...) { ; }
#define INTDBG(format, args...) { ; }
#define THRDBG(format, args...) { ; }
#define MPXDBG(format, args...) { ; }
#define OVFDBG(format, args...) { ; }
#define PRFDBG(format, args...) { ; }
#define MEMDBG(format, args...) { ; }
#define LEAKDBG(format, args...) { ; }
#define HLDBG(format, args...) { ; }
#define PAPIDEBUG(level, format, args...) { ; }
#endif
#endif

/*
 * Debug functions for platforms without vararg macro support
 */

#ifdef NO_VARARG_MACRO

static void PAPIDEBUG( int level, char *format, va_list args )
{
#ifdef DEBUG

	if ( ISLEVEL( level ) ) {
		vfprintf( stderr, format, args );
	} else
#endif
		return;
}

static void
_SUBDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_SUBSTRATE, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define SUBDBG do { \
 if (DEBUG_SUBSTRATE & _papi_hwi_debug) {\
   DEBUGLABEL( DEBUGLEVEL ( DEBUG_SUBSTRATE ) ); \
 } \
} while(0); _SUBDBG
#else
#define SUBDBG _SUBDBG
#endif

static void
_APIDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_API, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define APIDBG do { \
  if (DEBUG_API&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_API ) ); \
  } \
} while(0); _APIDBG
#else
#define APIDBG _APIDBG
#endif

static void
_INTDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_INTERNAL, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define INTDBG do { \
    if (DEBUG_INTERNAL&_papi_hwi_debug) {\
	  DEBUGLABEL( DEBUGLEVEL ( DEBUG_INTERNAL ) ); \
	} \
} while(0); _INTDBG
#else
#define INTDBG _INTDBG
#endif 

static void
_THRDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_THREADS, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define THRDBG do { \
  if (DEBUG_THREADS&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_THREADS ) ); \
  } \
} while(0); _THRDBG
#else
#define THRDBG _THRDBG
#endif

static void
_MPXDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_MULTIPLEX, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define MPXDBG do { \
  if (DEBUG_MULTIPLEX&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_MULTIPLEX ) ); \
  } \
} while(0); _MPXDBG
#else
#define MPXDBG _MPXDBG
#endif

static void
_OVFDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_OVERFLOW, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define OVFDBG do { \
  if (DEBUG_OVERFLOW&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_OVERFLOW ) ); \
  } \
} while(0); _OVFDBG
#else
#define OVFDBG _OVFDBG
#endif

static void
_PRFDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_PROFILE, format, args );
	va_end(args);
#endif
}
#ifdef DEBUG
#define PRFDBG do { \
  if (DEBUG_PROFILE&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_PROFILE ) ); \
  } \
} while(0); _PRFDBG
#else
#define PRFDBG _PRFDBG
#endif

static void
_MEMDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_MEMORY, format , args);
	va_end(args);
#endif
}
#ifdef DEBUG
#define MEMDBG do { \
  if (DEBUG_MEMORY&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_MEMORY ) ); \
  } \
} while(0); _MEMDBG
#else
#define MEMDBG _MEMDBG
#endif

static void
_LEAKDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_LEAK, format , args);
	va_end(args);
#endif
}
#ifdef DEBUG
#define LEAKDBG do { \
  if (DEBUG_LEAK&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_LEAK ) ); \
  } \
} while(0); _LEAKDBG
#else
#define LEAKDBG _LEAKDBG
#endif

static void
_HLDBG( char *format, ... )
{
#ifdef DEBUG
	va_list args;
	va_start(args, format);
	PAPIDEBUG( DEBUG_HIGHLEVEL, format , args);
	va_end(args);
#endif
}
#ifdef DEBUG
#define HLDBG do { \
  if (DEBUG_HIGHLEVEL&_papi_hwi_debug) {\
	DEBUGLABEL( DEBUGLEVEL ( DEBUG_HIGHLEVEL ) ); \
  } \
} while(0); _HLDBG
#else
#define HLDBG _HLDBG
#endif

/* ifdef NO_VARARG_MACRO */
#endif

#endif /* PAPI_DEBUG_H */
