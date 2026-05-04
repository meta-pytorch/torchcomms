/**
 * @file    lcuda_debug.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __LCUDA_DEBUG_H__
#define __LCUDA_DEBUG_H__

#include "papi.h"
#include "papi_internal.h"

/* Macro to either exit or continue depending on switch */
#define EXIT_OR_NOT
#ifdef EXIT_ON_ERROR
#   undef EXIT_OR_NOT
#   define EXIT_OR_NOT exit(-1)
#endif

/* Function calls */
#define COMPDBG(format, args...) SUBDBG("COMPDEBUG: " format, ## args);

/* General log */
#define LOGDBG(format, args...) SUBDBG("LOG: " format, ## args);

/* Lock and unlock calls */
#define LOCKDBG(format, args...) SUBDBG("LOCK: " format, ## args);

/* ERROR */
#define ERRDBG(format, args...) SUBDBG("ERROR: " format, ## args);

/* Log cuda driver and runtime calls */
#define LOGCUDACALL(format, args...) SUBDBG("CUDACALL: " format, ## args);

/* Log cupti and perfworks calls */
#define LOGCUPTICALL(format, args...) SUBDBG("CUPTICALL: " format, ## args);

#endif  /* __LCUDA_DEBUG_H__ */
