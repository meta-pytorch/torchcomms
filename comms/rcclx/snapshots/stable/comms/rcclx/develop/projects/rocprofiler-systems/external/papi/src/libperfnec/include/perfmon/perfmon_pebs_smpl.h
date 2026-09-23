/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of version 2 of the GNU General Public
 * License as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 *
 */
#ifndef __PERFMON_PEBS_SMPL_H__
#define __PERFMON_PEBS_SMPL_H__ 1

/*
 * The 32-bit and 64-bit formats are identical, thus we use only
 * one name for the format.
 */
#define PFM_PEBS_SMPL_NAME	"pebs"

#define PFM_PEBS_NUM_CNT_RESET	8
/*
 * format specific parameters (passed at context creation)
 *
 * intr_thres: index from start of buffer of entry where the
 * PMU interrupt must be triggered. It must be several samples
 * short of the end of the buffer.
 */
typedef struct {
	uint64_t buf_size;		/* size of the PEBS buffer in bytes */
	uint64_t cnt_reset[PFM_PEBS_NUM_CNT_RESET];/* counter reset values */
	uint64_t reserved2[23];		/* for future use */
} pfm_pebs_smpl_arg_t;

/*
 * This header is at the beginning of the sampling buffer returned to the user.
 *
 * Because of PEBS alignement constraints, the actual PEBS buffer area does
 * not necessarily begin right after the header. The hdr_start_offs must be
 * used to compute the first byte of the buffer. The offset is defined as
 * the number of bytes between the end of the header and the beginning of
 * the buffer. As such the formula is:
 * 	actual_buffer = (unsigned long)(hdr+1)+hdr->hdr_start_offs
 */
typedef struct {
	uint64_t overflows;		/* #overflows for buffer */
	uint64_t count;			/* number of valid samples */
	uint64_t buf_size;		/* total buffer size */
	uint64_t pebs_size;		/* pebs buffer size */
	uint32_t version;		/* smpl format version */
	uint32_t entry_size;		/* pebs sample size */
	uint64_t reserved2[11];		/* for future use */
} pfm_pebs_smpl_hdr_t;

/*
 * Sample format as mandated by Intel documentation.
 * The same format is used in both 32 and 64 bit modes.
 */
typedef struct {
	uint64_t	eflags;
	uint64_t	ip;
	uint64_t	eax;
	uint64_t	ebx;
	uint64_t	ecx;
	uint64_t	edx;
	uint64_t	esi;
	uint64_t	edi;
	uint64_t	ebp;
	uint64_t	esp;
	uint64_t	r8;	/* 0 in 32-bit mode */
	uint64_t	r9;	/* 0 in 32-bit mode */
	uint64_t	r10;	/* 0 in 32-bit mode */
	uint64_t	r11;	/* 0 in 32-bit mode */
	uint64_t	r12;	/* 0 in 32-bit mode */
	uint64_t	r13;	/* 0 in 32-bit mode */
	uint64_t	r14;	/* 0 in 32-bit mode */
	uint64_t	r15;	/* 0 in 32-bit mode */
} pfm_pebs_core_smpl_entry_t; 

/*
 * Sample format as mandated by Intel documentation.
 * The same format is used in both 32 and 64 bit modes.
 */
typedef struct {
	uint64_t	eflags;
	uint64_t	ip;
	uint64_t	eax;
	uint64_t	ebx;
	uint64_t	ecx;
	uint64_t	edx;
	uint64_t	esi;
	uint64_t	edi;
	uint64_t	ebp;
	uint64_t	esp;
	uint64_t	r8;	/* 0 in 32-bit mode */
	uint64_t	r9;	/* 0 in 32-bit mode */
	uint64_t	r10;	/* 0 in 32-bit mode */
	uint64_t	r11;	/* 0 in 32-bit mode */
	uint64_t	r12;	/* 0 in 32-bit mode */
	uint64_t	r13;	/* 0 in 32-bit mode */
	uint64_t	r14;	/* 0 in 32-bit mode */
	uint64_t	r15;	/* 0 in 32-bit mode */
	uint64_t	ia32_perf_global_status;
	uint64_t	daddr;
	uint64_t	dsrc_enc;
	uint64_t	latency;
} pfm_pebs_nhm_smpl_entry_t; 

/*
 * 64-bit PEBS record format is described in
 * http://www.intel.com/technology/64bitextensions/30083502.pdf
 *
 * The format does not peek at samples. The sample structure is only
 * used to ensure that the buffer is large enough to accomodate one
 * sample.
 */
#ifdef __i386__
typedef struct {
	uint32_t	eflags;
	uint32_t	ip;
	uint32_t	eax;
	uint32_t	ebx;
	uint32_t	ecx;
	uint32_t	edx;
	uint32_t	esi;
	uint32_t	edi;
	uint32_t	ebp;
	uint32_t	esp;
} pfm_pebs_p4_smpl_entry_t;
#else
typedef struct {
	uint64_t	eflags;
	uint64_t	ip;
	uint64_t	eax;
	uint64_t	ebx;
	uint64_t	ecx;
	uint64_t	edx;
	uint64_t	esi;
	uint64_t	edi;
	uint64_t	ebp;
	uint64_t	esp;
	uint64_t	r8;
	uint64_t	r9;
	uint64_t	r10;
	uint64_t	r11;
	uint64_t	r12;
	uint64_t	r13;
	uint64_t	r14;
	uint64_t	r15;
} pfm_pebs_p4_smpl_entry_t;
#endif

#define PFM_PEBS_SMPL_VERSION_MAJ 1U
#define PFM_PEBS_SMPL_VERSION_MIN 0U
#define PFM_PEBS_SMPL_VERSION (((PFM_PEBS_SMPL_VERSION_MAJ&0xffff)<<16)|\
				   (PFM_PEBS_SMPL_VERSION_MIN & 0xffff))

#endif /* __PERFMON_PEBS_SMPL_H__ */
