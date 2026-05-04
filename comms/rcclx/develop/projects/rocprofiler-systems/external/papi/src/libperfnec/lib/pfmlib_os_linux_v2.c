/*
 * pfmlib_os_linux_v2.c: Perfmon2 syscall API
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
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <syscall.h>
#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "pfmlib_priv.h"

/*
 * v2.x interface
 */
#define PFM_pfm_create_context		(_pfmlib_get_sys_base()+0)
#define PFM_pfm_write_pmcs		(_pfmlib_get_sys_base()+1)
#define PFM_pfm_write_pmds		(_pfmlib_get_sys_base()+2)
#define PFM_pfm_read_pmds		(_pfmlib_get_sys_base()+3)
#define PFM_pfm_load_context		(_pfmlib_get_sys_base()+4)
#define PFM_pfm_start			(_pfmlib_get_sys_base()+5)
#define PFM_pfm_stop			(_pfmlib_get_sys_base()+6)
#define PFM_pfm_restart			(_pfmlib_get_sys_base()+7)
#define PFM_pfm_create_evtsets		(_pfmlib_get_sys_base()+8)
#define PFM_pfm_getinfo_evtsets		(_pfmlib_get_sys_base()+9)
#define PFM_pfm_delete_evtsets		(_pfmlib_get_sys_base()+10)
#define PFM_pfm_unload_context		(_pfmlib_get_sys_base()+11)

/*
 * argument to v2.2 pfm_create_context()
 * ALWAYS use pfarg_ctx_t in programs, libpfm
 * does convert ths structure on the fly if v2.2. is detected
 */
typedef struct {
	unsigned char	ctx_smpl_buf_id[16];	/* which buffer format to use */
	uint32_t	ctx_flags;		/* noblock/block/syswide */
	int32_t		ctx_fd;			/* ret arg: fd for context */
	uint64_t	ctx_smpl_buf_size;	/* ret arg: actual buffer sz */
	uint64_t	ctx_reserved3[12];	/* for future use */
} pfarg_ctx22_t;

/*
 * perfmon2 compatibility layer with perfmon3
 */
#ifndef PFMLIB_OLD_PFMV2
static int
pfm_create_context_2v3(pfarg_ctx_t *ctx, char *name, void *smpl_arg, size_t smpl_size)
{
	pfarg_sinfo_t cinfo;
	uint32_t fl;

	/*
 	 * simulate kernel returning error on NULL ctx
 	 */
	if (!ctx) {
		errno = EINVAL;
		return -1;
	} 

	/*
 	 * if sampling format is used, then force SMPL_FMT
 	 * and PFM_FL_SINFO because it comes first
 	 */
	fl = ctx->ctx_flags;
	if (name || smpl_arg || smpl_size)
		fl |= PFM_FL_SMPL_FMT;
	
	return pfm_create(fl, &cinfo, name, smpl_arg, smpl_size);
}

static int
pfm_write_pmcs_2v3(int fd, pfarg_pmc_t *pmcs, int count)
{
	pfarg_pmr_t *pmrs;
	int errno_save;
	int i, ret;
	size_t sz;

	sz = count * sizeof(pfarg_pmr_t);

	if (!pmcs)
		return pfm_write(fd, 0, PFM_RW_PMC, NULL, sz);

	pmrs = calloc(count, sizeof(*pmrs));
	if (!pmrs) {
		errno = ENOMEM;
		return -1;
	}

	for (i=0 ; i < count; i++) {
		pmrs[i].reg_num = pmcs[i].reg_num;
		pmrs[i].reg_set = pmcs[i].reg_set;
		pmrs[i].reg_flags = pmcs[i].reg_flags;
		pmrs[i].reg_value = pmcs[i].reg_value;
	}

	ret = pfm_write(fd, 0, PFM_RW_PMC, pmrs, sz);
	errno_save = errno;
	free(pmrs);
	errno = errno_save;
	return ret;
}

static int
pfm_write_pmds_2v3(int fd, pfarg_pmd_t *pmds, int count)
{
	pfarg_pmd_attr_t *pmas;
	size_t sz;
	int errno_save;
	int i, ret;

	sz = count * sizeof(*pmas);

	if (!pmds)
		return pfm_write(fd, 0, PFM_RW_PMD, NULL, sz);

	pmas = calloc(count, sizeof(*pmas));
	if (!pmas) {
		errno = ENOMEM;
		return -1;
	}

	for (i=0 ; i < count; i++) {
		pmas[i].reg_num = pmds[i].reg_num;
		pmas[i].reg_set = pmds[i].reg_set;
		pmas[i].reg_flags = pmds[i].reg_flags;
		pmas[i].reg_value = pmds[i].reg_value;

		pmas[i].reg_long_reset = pmds[i].reg_long_reset;
		pmas[i].reg_short_reset = pmds[i].reg_short_reset;
		/* skip last_value not used on write */

		pmas[i].reg_ovfl_swcnt = pmds[i].reg_ovfl_switch_cnt;

		memcpy(pmas[i].reg_smpl_pmds, pmds[i].reg_smpl_pmds, sizeof(pmds[i].reg_smpl_pmds));
		memcpy(pmas[i].reg_reset_pmds, pmds[i].reg_reset_pmds, sizeof(pmds[i].reg_reset_pmds));

		pmas[i].reg_smpl_eventid = pmds[i].reg_smpl_eventid;
		pmas[i].reg_random_mask = pmds[i].reg_random_mask;
	}

	ret = pfm_write(fd, 0, PFM_RW_PMD_ATTR, pmas, sz);

	errno_save = errno;
	free(pmas);
	errno = errno_save;

	return ret;
}

static int
pfm_read_pmds_2v3(int fd, pfarg_pmd_t *pmds, int count)
{
	pfarg_pmd_attr_t *pmas;
	int errno_save;
	int i, ret;
	size_t sz;

	sz = count * sizeof(*pmas);

	if (!pmds)
		return pfm_write(fd, 0, PFM_RW_PMD, NULL, sz);

	pmas = calloc(count, sizeof(*pmas));
	if (!pmas) {
		errno = ENOMEM;
		return -1;
	}

	for (i=0 ; i < count; i++) {
		pmas[i].reg_num = pmds[i].reg_num;
		pmas[i].reg_set = pmds[i].reg_set;
		pmas[i].reg_flags = pmds[i].reg_flags;
		pmas[i].reg_value = pmds[i].reg_value;
	}

	ret = pfm_read(fd, 0, PFM_RW_PMD_ATTR, pmas, sz);

	errno_save = errno;

	for (i=0 ; i < count; i++) {
		pmds[i].reg_value = pmas[i].reg_value;

		pmds[i].reg_long_reset = pmas[i].reg_long_reset;
		pmds[i].reg_short_reset = pmas[i].reg_short_reset;
		pmds[i].reg_last_reset_val = pmas[i].reg_last_value;

		pmds[i].reg_ovfl_switch_cnt = pmas[i].reg_ovfl_swcnt;
		/* skip reg_smpl_pmds */
		/* skip reg_reset_pmds */
		/* skip reg_smpl_eventid */
		/* skip reg_random_mask */
	}
	free(pmas);
	errno = errno_save;
	return ret;
}

static int
pfm_load_context_2v3(int fd, pfarg_load_t *load)
{
	if (!load) {
		errno = EINVAL;
		return -1;
	}
	return pfm_attach(fd, 0, load->load_pid);
}

static int
pfm_start_2v3(int fd, pfarg_start_t *start)
{
	if (start) {
		__pfm_vbprintf("pfarg_start_t not supported in v3.x\n");
		errno = EINVAL;
		return -1;
	}
	return pfm_set_state(fd, 0, PFM_ST_START);
}
static int
pfm_stop_2v3(int fd)
{
	return pfm_set_state(fd, 0, PFM_ST_STOP);
}

static int
pfm_restart_2v3(int fd)
{
	return pfm_set_state(fd, 0, PFM_ST_RESTART);
}

static int
pfm_create_evtsets_2v3(int fd, pfarg_setdesc_t *setd, int count)
{
	/* set_desc an setdesc are identical so we can cast */
	return pfm_create_sets(fd, 0, (pfarg_set_desc_t *)setd, count * sizeof(pfarg_setdesc_t));
}

static int
pfm_delete_evtsets_2v3(int fd, pfarg_setdesc_t *setd, int count)
{
	__pfm_vbprintf("pfm_delete_evtsets not supported in v3.x\n");
	errno = EINVAL;
	return -1;
}

static int
pfm_getinfo_evtsets_2v3(int fd, pfarg_setinfo_t *info, int count)
{
	pfarg_sinfo_t cinfo;
	pfarg_set_info_t *sif;
	int fdx, i, ret, errno_save;

	if (!info) {
		errno = EFAULT;
		return -1;
	}
	/*
	 * initialize bitmask to all available and defer checking
	 * until kernel. That means libpfm must be misled but we
	 * have no other way of fixing this
	 */
	memset(&cinfo, -1, sizeof(cinfo));

	/*
	 * XXX: relies on the fact that cinfo is independent
	 * of the session type (which is wrong in the future)
	 */
	fdx = pfm_create(0, &cinfo);
	if (fdx > -1)
		close(fdx);

	sif = calloc(count, sizeof(*sif));
	if (!sif) {
		errno = ENOMEM;
		return -1;
	}

	for (i=0 ; i < count; i++)
		sif[i].set_id = info[i].set_id;

	ret = pfm_getinfo_sets(fd, 0, sif, count * sizeof(pfarg_set_info_t));
	errno_save = errno;
	if (ret)
		goto skip;

	for (i=0 ; i < count; i++) {
		info[i].set_flags = 0;

		memcpy(info[i].set_ovfl_pmds,
		       sif[i].set_ovfl_pmds,
		       sizeof(info[i].set_ovfl_pmds));

		info[i].set_runs = sif[i].set_runs;
		info[i].set_timeout = sif[i].set_timeout;
		info[i].set_act_duration = sif[i].set_duration;

		memcpy(info[i].set_avail_pmcs,
		       cinfo.sif_avail_pmcs,
		       sizeof(info[i].set_avail_pmcs));

		memcpy(info[i].set_avail_pmds,
		       cinfo.sif_avail_pmds,
		       sizeof(info[i].set_avail_pmds));
	}
skip:
	free(sif);
	errno = errno_save;
	return ret;
}

static int
pfm_unload_context_2v3(int fd)
{
	return pfm_attach(fd, 0, PFM_NO_TARGET);
}

#else /* PFMLIB_OLD_PFMV2 */

static int
pfm_create_context_2v3(pfarg_ctx_t *ctx, char *name, void *smpl_arg, size_t smpl_size)
{
	return -1;
}

static int
pfm_write_pmcs_2v3(int fd, pfarg_pmc_t *pmcs, int count)
{
	return -1;
}

static int
pfm_write_pmds_2v3(int fd, pfarg_pmd_t *pmds, int count)
{
	return -1;
}

static int
pfm_read_pmds_2v3(int fd, pfarg_pmd_t *pmds, int count)
{
	return -1;
}

static int
pfm_load_context_2v3(int fd, pfarg_load_t *load)
{
	return -1;
}

static int
pfm_start_2v3(int fd, pfarg_start_t *start)
{
	return -1;
}
static int
pfm_stop_2v3(int fd)
{
	return -1;
}

static int
pfm_restart_2v3(int fd)
{
	return -1;
}

static int
pfm_create_evtsets_2v3(int fd, pfarg_setdesc_t *setd, int count)
{
	return -1;
}

static int
pfm_delete_evtsets_2v3(int fd, pfarg_setdesc_t *setd, int count)
{
	return -1;
}

static int
pfm_getinfo_evtsets_2v3(int fd, pfarg_setinfo_t *info, int count)
{
	return -1;
}

static int
pfm_unload_context_2v3(int fd)
{
	return -1;
}
#endif /* PFMLIB_OLD_PFMV2 */

int
pfm_load_context(int fd, pfarg_load_t *load)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_load_context, fd, load);
	return pfm_load_context_2v3(fd, load);
}

int
pfm_start(int fd, pfarg_start_t *start)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_start, fd, start);
	return pfm_start_2v3(fd, start);
}

int
pfm_stop(int fd)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_stop, fd);
	return pfm_stop_2v3(fd);
}

int
pfm_restart(int fd)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_restart, fd);
	return pfm_restart_2v3(fd);
}

int
pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_create_evtsets, fd, setd, count);
	return pfm_create_evtsets_2v3(fd, setd, count);
}

int
pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_delete_evtsets, fd, setd, count);
	return pfm_delete_evtsets_2v3(fd, setd, count);
}

int
pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_getinfo_evtsets, fd, info, count);

	return pfm_getinfo_evtsets_2v3(fd, info, count);
}

int
pfm_unload_context(int fd)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_unload_context, fd);

	return pfm_unload_context_2v3(fd);
}

int
pfm_create_context(pfarg_ctx_t *ctx, char *name, void *smpl_arg, size_t smpl_size)
{
	if (_pfmlib_major_version < 3) {
		/*
		 * In perfmon v2.2, the pfm_create_context() call had a
		 * different return value. It used to return errno, in v2.3
		 * it returns the file descriptor.
		 */
		if (_pfmlib_minor_version < 3) {
			int r;
			pfarg_ctx22_t ctx22;

			/* transfer the v2.3 contents to v2.2 for sys call */
			memset (&ctx22, 0, sizeof(ctx22));
			if (name != NULL) {
				memcpy (ctx22.ctx_smpl_buf_id, name, 16);
			}
			ctx22.ctx_flags = ctx->ctx_flags;
			/* ctx22.ctx_fd returned */
			/* ctx22.ctx_smpl_buf_size returned */
			memcpy (ctx22.ctx_reserved3, &ctx->ctx_reserved1, 64);

			r = syscall (PFM_pfm_create_context, &ctx22, smpl_arg, smpl_size);

			/* transfer the v2.2 contents back to v2.3 */
			ctx->ctx_flags = ctx22.ctx_flags;
			memcpy (&ctx->ctx_reserved1, ctx22.ctx_reserved3, 64);

			return (r < 0 ? r : ctx22.ctx_fd);
		} else {
			return (int)syscall(PFM_pfm_create_context, ctx, name, smpl_arg, smpl_size);
		}
	}
	return pfm_create_context_2v3(ctx, name, smpl_arg, smpl_size);
}

int
pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_write_pmcs, fd, pmcs, count);
	return pfm_write_pmcs_2v3(fd, pmcs, count);
}

int
pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_write_pmds, fd, pmds, count);
	return pfm_write_pmds_2v3(fd, pmds, count);
}

int
pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
	if (_pfmlib_major_version < 3)
		return (int)syscall(PFM_pfm_read_pmds, fd, pmds, count);
	return pfm_read_pmds_2v3(fd, pmds, count);
}

#ifdef __ia64__
#define __PFMLIB_OS_COMPILE
#include <perfmon/pfmlib.h>

/*
 * this is the old perfmon2 interface, maintained for backward
 * compatibility reasons with older applications. This is for IA-64 ONLY.
 */
int
perfmonctl(int fd, int cmd, void *arg, int narg)
{
	return syscall(__NR_perfmonctl, fd, cmd, arg, narg);
}
#endif /* __ia64__ */
