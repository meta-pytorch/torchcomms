/*
 * Copyright (c) 2007 Cray Inc.
 * Contributed by Steve Kaufmann <sbk@cray.com> based on code from
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
 */
#ifndef __PFMLIB_OS_CRAYX2_H__
#define __PFMLIB_OS_CRAYX2_H__

#ifndef __PFMLIB_OS_H__
#error "you should never include this file directly, use pfmlib_os.h"
#endif

#include <perfmon/perfmon.h>

/*
 * macros version of pfm_self_start/pfm_self_stop to be used in per-process self-monitoring sessions.
 * they are also defined as real functions.
 *
 * DO NOT USE on system-wide sessions.
 */
static inline int
pfm_self_start(int fd)
{
	return pfm_start(fd, NULL);
}

static inline int
pfm_self_stop(int fd)
{
	return pfm_stop(fd);
}

#endif /* __PFMLIB_OS_CRAYX2_H__ */
