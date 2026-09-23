/*
 * pfmlib_os_linux_v3.c: Perfmon3 API syscalls
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <syscall.h>
#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "pfmlib_priv.h"

/*
 * v3.x interface
 */
#define PFM_pfm_create			(_pfmlib_get_sys_base()+0)
#define PFM_pfm_write			(_pfmlib_get_sys_base()+1)
#define PFM_pfm_read			(_pfmlib_get_sys_base()+2)
#define PFM_pfm_attach			(_pfmlib_get_sys_base()+3)
#define PFM_pfm_set_state		(_pfmlib_get_sys_base()+4)
#define PFM_pfm_create_sets		(_pfmlib_get_sys_base()+5)
#define PFM_pfm_getinfo_sets		(_pfmlib_get_sys_base()+6)

/*
 * perfmon v3 interface
 */
int
//pfm_create(int flags, pfarg_sinfo_t *sif, char *name, void *smpl_arg, size_t smpl_size)
pfm_create(int flags, pfarg_sinfo_t *sif, ...)
{
	va_list ap;
	char *name = NULL;
	void *smpl_arg = NULL;
	size_t smpl_size = 0;
	int ret;

	if (_pfmlib_major_version < 3) {
		errno = ENOSYS;
		return -1;
	}

	if (flags & PFM_FL_SMPL_FMT)
		va_start(ap, sif);
	
	if (flags & PFM_FL_SMPL_FMT) {
		name = va_arg(ap, char *);
		smpl_arg = va_arg(ap, void *);
		smpl_size = va_arg(ap, size_t);
	}

	ret = (int)syscall(PFM_pfm_create, flags, sif, name, smpl_arg, smpl_size);

	if (flags & PFM_FL_SMPL_FMT)
		va_end(ap);

	return ret;
}

int
pfm_write(int fd, int flags, int type, void *pms, size_t sz)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_write, fd, flags, type, pms, sz);
}

int
pfm_read(int fd, int flags, int type, void *pms, size_t sz)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_read, fd, flags, type, pms, sz);
}

int
pfm_create_sets(int fd, int flags, pfarg_set_desc_t *setd, size_t sz)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_create_sets, fd, flags, setd, sz);
}

int
pfm_getinfo_sets(int fd, int flags, pfarg_set_info_t *info, size_t sz)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_getinfo_sets, fd, flags, info, sz);
}

int
pfm_attach(int fd, int flags, int target)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_attach, fd, flags, target);
}

int
pfm_set_state(int fd, int flags, int state)
{
	if (_pfmlib_major_version < 3)
		return -ENOSYS;
	return (int)syscall(PFM_pfm_set_state, fd, flags, state);
}
