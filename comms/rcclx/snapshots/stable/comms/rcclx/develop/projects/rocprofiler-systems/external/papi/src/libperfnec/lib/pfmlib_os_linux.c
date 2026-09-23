/*
 * pfmlib_os.c: set of functions OS dependent functions
 *
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/utsname.h>
#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "pfmlib_priv.h"

int _pfmlib_sys_base; /* syscall base */
int _pfmlib_major_version; /* kernel perfmon major version */
int _pfmlib_minor_version; /* kernel perfmon minor version */

/*
 * helper function to retrieve one value from /proc/cpuinfo
 * for internal libpfm use only
 * attr: the attribute (line) to look for
 * ret_buf: a buffer to store the value of the attribute (as a string)
 * maxlen : number of bytes of capacity in ret_buf
 *
 * ret_buf is null terminated.
 *
 * Return:
 * 	0 : attribute found, ret_buf populated
 * 	-1: attribute not found
 */
int
__pfm_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{

	int ret = 0;
	return ret;
}

static void adjust__pfmlib_sys_base(int version)
{
}

static void
pfm_init_syscalls_hardcoded(void)
{
}

static int
pfm_init_syscalls_sysfs(void)
{
}
static int
pfm_init_version_sysfs(void)
{
}


void
pfm_init_syscalls(void)
{
}
