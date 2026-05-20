/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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
 * applications on Linux.
 */
#ifndef __PFMLIB_I386_P6_PRIV_H__
#define __PFMLIB_I386_P6_PRIV_H__

#define PFMLIB_I386_P6_MAX_UMASK	16

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
} pme_i386_p6_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	pme_i386_p6_umask_t	pme_umasks[PFMLIB_I386_P6_MAX_UMASK]; /* umask desc */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
} pme_i386_p6_entry_t;

/*
 * pme_flags values
 */
#define PFMLIB_I386_P6_UMASK_COMBO	0x01 /* unit mask can be combined */
#define PFMLIB_I386_P6_CTR0_ONLY	0x02 /* event can only be counted on counter 0 */
#define PFMLIB_I386_P6_CTR1_ONLY	0x04 /* event can only be counted on counter 1 */

#endif /* __PFMLIB_I386_P6_PRIV_H__ */
