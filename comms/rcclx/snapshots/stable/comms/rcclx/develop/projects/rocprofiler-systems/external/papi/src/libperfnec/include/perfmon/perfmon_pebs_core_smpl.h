/*
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
 *               Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file implements the sampling format to support Intel
 * Precise Event Based Sampling (PEBS) feature of Intel
 * Core and Atom processors.
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
 *  format does not change between 32-bit and 64-bit modes unlike on the
 *  Pentium 4 version of PEBS.
 *
 *  What does the format do?
 *   It provides access to the PEBS feature for both 32-bit and 64-bit
 *   processors that support it.
 *
 *   The same code and data structures are used for both 32-bit and 64-bi
 *   modes. A single format name is used for both modes. In 32-bit mode,
 *   some of the extended registers are written to zero in each sample.
 *
 *   It is important to realize that the format provides a zero-copy
 *   environment for the samples, i.e,, the OS never touches the
 *   samples. Whatever the processor write is directly accessible to
 *   the user.
 *
 *   Parameters to the buffer can be passed via pfm_create_context() in
 *   the pfm_pebs_smpl_arg structure.
 */
#ifndef __PERFMON_PEBS_CORE_SMPL_H__
#define __PERFMON_PEBS_CORE_SMPL_H__ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <perfmon/perfmon.h>

#define PFM_PEBS_CORE_SMPL_NAME	"pebs_core"

/*
 * format specific parameters (passed at context creation)
 */
typedef struct {
	uint64_t	cnt_reset;	/* counter reset value */
	uint64_t	buf_size;	/* size of the buffer in bytes */
	uint64_t	intr_thres;	/* index of interrupt threshold entry */
	uint64_t	reserved[6];	/* for future use */
} pfm_pebs_core_smpl_arg_t;

/*
 * DS Save Area
 */
typedef struct {
	uint64_t	bts_buf_base;
	uint64_t	bts_index;
	uint64_t	bts_abs_max;
	uint64_t	bts_intr_thres;
	uint64_t	pebs_buf_base;
	uint64_t	pebs_index;
	uint64_t	pebs_abs_max;
	uint64_t	pebs_intr_thres;
	uint64_t	pebs_cnt_reset;
} pfm_ds_area_core_t; 

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
	pfm_ds_area_core_t	ds;		/* DS management Area */
} pfm_pebs_core_smpl_hdr_t;

/*
 * PEBS record format as for both 32-bit and 64-bit modes
 */
typedef struct {
	uint64_t eflags;
	uint64_t ip;
	uint64_t eax;
	uint64_t ebx;
	uint64_t ecx;
	uint64_t edx;
	uint64_t esi;
	uint64_t edi;
	uint64_t ebp;
	uint64_t esp;
	uint64_t r8;	/* 0 in 32-bit mode */
	uint64_t r9;	/* 0 in 32-bit mode */
	uint64_t r10;	/* 0 in 32-bit mode */
	uint64_t r11;	/* 0 in 32-bit mode */
	uint64_t r12;	/* 0 in 32-bit mode */
	uint64_t r13;	/* 0 in 32-bit mode */
	uint64_t r14;	/* 0 in 32-bit mode */
	uint64_t r15;	/* 0 in 32-bit mode */
} pfm_pebs_core_smpl_entry_t;

#define PFM_PEBS_CORE_SMPL_VERSION_MAJ 1U
#define PFM_PEBS_CORE_SMPL_VERSION_MIN 0U
#define PFM_PEBS_CORE_SMPL_VERSION (((PFM_PEBS_CORE_SMPL_VERSION_MAJ&0xffff)<<16)|\
				   (PFM_PEBS_CORE_SMPL_VERSION_MIN & 0xffff))

#ifdef __cplusplus
};
#endif

#endif /* __PERFMON_PEBS_CORE_SMPL_H__ */
