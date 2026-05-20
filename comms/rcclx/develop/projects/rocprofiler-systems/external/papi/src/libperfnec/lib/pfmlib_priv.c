/*
 * pfmlib_priv.c: set of internal utility functions for all architectures
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"

/*
 * file for all libpfm verbose and debug output
 *
 * By default, it is set to stderr, unless the
 * PFMLIB_DEBUG_STDOUT environment variable is set
 */
FILE *libpfm_fp;

/*
 * by convention all internal utility function must be prefixed by __
 */

/*
 * debug printf
 */
void
__pfm_vbprintf(const char *fmt, ...)
{
	va_list ap;

	if (pfm_config.options.pfm_verbose == 0)
		return;

	va_start(ap, fmt);
	vfprintf(libpfm_fp, fmt, ap);
	va_end(ap);
}

int
__pfm_check_event(pfmlib_event_t *e)
{
	unsigned int n, j;

	if (e->event >= pfm_current->pme_count)
		return PFMLIB_ERR_INVAL;

	n = pfm_num_masks(e->event);
	if (n == 0 && e->num_masks)
		return PFMLIB_ERR_UMASK;

	for(j=0; j < e->num_masks; j++) {
		if (e->unit_masks[j] >= n)
			return PFMLIB_ERR_UMASK;
	}
	/*
	 * if event has umask, but non specified by user, then
	 * return:
	 *   - error if no default umask is defined
	 *   - success if default umask exists for event
	 */
	if (n && j == 0) {
		if (pfm_current->has_umask_default
		    && pfm_current->has_umask_default(e->event))
			return PFMLIB_SUCCESS;
		return PFMLIB_ERR_UMASK;
	}
	return PFMLIB_SUCCESS;
}
