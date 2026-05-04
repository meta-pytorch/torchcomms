/*
 * Contributed by Philip Mucci <mucci@cs.utk.edu> based on code from
 * Copyright (c) 2004-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
 */
#ifndef __PFMLIB_SICORTEX_PRIV_H__
#define __PFMLIB_SICORTEX_PRIV_H__
#include "pfmlib_gen_mips64_priv.h"

#define PFMLIB_SICORTEX_MAX_UMASK 5

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
} pme_sicortex_umask_t;

typedef struct {
	char			*pme_name;
	char			*pme_desc; /* text description of the event */
	unsigned int		pme_code;  /* event mask, holds room for four events, low 8 bits cntr0, ... high 8 bits cntr3 */
	unsigned int		pme_counters;   /* Which counter event lives on */
	unsigned int		pme_numasks;	/* number of umasks */
	pme_sicortex_umask_t pme_umasks[PFMLIB_SICORTEX_MAX_UMASK]; /* umask desc */
} pme_sicortex_entry_t;
  
static pme_sicortex_umask_t sicortex_scb_umasks[PFMLIB_SICORTEX_MAX_UMASK] = {
  {
    "IFOTHER_NONE","Both buckets count independently",0x00
  },
  {
    "IFOTHER_AND","Increment where this event counts and the opposite bucket counts",0x02
  },
  {
    "IFOTHER_ANDNOT","Increment where this event counts and the opposite bucket does not",0x04
  },
  {
    "HIST_NONE","Count cycles where the event is asserted",0x0
  },
  {
    "HIST_EDGE","Histogram on edges of the specified event",0x1
  }
};
#endif /* __PFMLIB_GEN_MIPS64_PRIV_H__ */
