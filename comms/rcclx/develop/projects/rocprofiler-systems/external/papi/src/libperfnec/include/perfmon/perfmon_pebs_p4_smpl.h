/*
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
 *               Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
 * This file implements the sampling format to support Intel
 * Precise Event Based Sampling (PEBS) feature of Pentium 4
 * and other Netburst-based processors. Not to be used for
 * Intel Core-based processors.
 *
 * What is PEBS?
 * ------------
 *  This is a hardware feature to enhance sampling by providing
 *  better precision as to where a sample is taken. This avoids the
 *  typical skew in the instruction one can observe with any
 *  interrupt-based sampling technique.
 *
 *  PEBS also lowers sampling overhead significantly by having the
 *  processor store samples instead of the OS. PMU interrupt are only
 *  generated after multiple samples are written.
 *
 *  Another benefit of PEBS is that samples can be captured inside
 *  critical sections where interrupts are masked.
 *
 * How does it work?
 *  PEBS effectively implements a Hw buffer. The Os must pass a region
 *  of memory where samples are to be stored. The region can have any
 *  size. The OS must also specify the sampling period to reload. The PMU
 *  will interrupt when it reaches the end of the buffer or a specified
 *  threshold location inside the memory region.
 *
 *  The description of the buffer is stored in the Data Save Area (DS).
 *  The samples are stored sequentially in the buffer. The format of the
 *  buffer is fixed and specified in the PEBS documentation.  The sample
 *  format changes between 32-bit and 64-bit modes due to extended register
 *  file.
 *
 *  PEBS does not work when HyperThreading is enabled due to certain MSR
 *  being shared being to two threads.
 *
 *  What does the format do?
 *   It provides access to the PEBS feature for both 32-bit and 64-bit
 *   processors that support it.
 *
 *   The same code is used for both 32-bit and 64-bit modes, but different
 *   format names are used because the two modes are not compatible due to
 *   data model and register file differences. Similarly the public data
 *   structures describing the samples are different.
 *
 *   It is important to realize that the format provides a zero-copy environment
 *   for the samples, i.e,, the OS never touches the samples. Whatever the
 *   processor write is directly accessible to the user.
 *
 *   Parameters to the buffer can be passed via pfm_create_context() in
 *   the pfm_pebs_smpl_arg structure.
 *
 *   It is not possible to mix a 32-bit PEBS application on top of a 64-bit
 *   host kernel.
 */
#ifndef __PERFMON_PEBS_P4_SMPL_H__
#define __PERFMON_PEBS_P4_SMPL_H__ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <perfmon/perfmon.h>

#ifdef __i386__
#define PFM_PEBS_P4_SMPL_NAME	"pebs32_p4"
#else
#define PFM_PEBS_P4_SMPL_NAME	"pebs64_p4"
#endif

/*
 * format specific parameters (passed at context creation)
 */
typedef struct {
	uint64_t	cnt_reset;	/* counter reset value */
	size_t		buf_size;	/* size of the buffer in bytes */
	size_t		intr_thres;	/* index of interrupt threshold entry */
	uint64_t	reserved[6];	/* for future use */
} pfm_pebs_p4_smpl_arg_t;

/*
 * DS Save Area as described in section 15.10.5
 */
typedef struct {
	unsigned long	bts_buf_base;
	unsigned long	bts_index;
	unsigned long	bts_abs_max;
	unsigned long	bts_intr_thres;
	unsigned long	pebs_buf_base;
	unsigned long	pebs_index;
	unsigned long	pebs_abs_max;
	unsigned long	pebs_intr_thres;
	uint64_t	pebs_cnt_reset;
} pfm_ds_area_p4_t; 

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
	uint64_t		overflows;	/* #overflows for buffer */
	size_t			buf_size;	/* bytes in the buffer */
	size_t			start_offs;	/* actual buffer start offset */
	uint32_t		version;	/* smpl format version */
	uint32_t		reserved1;	/* for future use */
	uint64_t		reserved2[5];	/* for future use */
	pfm_ds_area_p4_t	ds;		/* DS management Area */
} pfm_pebs_p4_smpl_hdr_t;

/*
 * PEBS record format as for both 32-bit and 64-bit modes
 */
typedef struct {
	unsigned long eflags;
	unsigned long ip;
	unsigned long eax;
	unsigned long ebx;
	unsigned long ecx;
	unsigned long edx;
	unsigned long esi;
	unsigned long edi;
	unsigned long ebp;
	unsigned long esp;
#ifdef __x86_64__
	unsigned long r8;
	unsigned long r9;
	unsigned long r10;
	unsigned long r11;
	unsigned long r12;
	unsigned long r13;
	unsigned long r14;
	unsigned long r15;
#endif
} pfm_pebs_p4_smpl_entry_t;

#define PFM_PEBS_P4_SMPL_VERSION_MAJ 1U
#define PFM_PEBS_P4_SMPL_VERSION_MIN 0U
#define PFM_PEBS_P4_SMPL_VERSION (((PFM_PEBS_P4_SMPL_VERSION_MAJ&0xffff)<<16)|\
				   (PFM_PEBS_P4_SMPL_VERSION_MIN & 0xffff))

#ifdef __cplusplus
};
#endif

#endif /* __PERFMON_PEBS_P4_SMPL_H__ */
