/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 *               Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file implements the new dfl sampling buffer format
 * for perfmon2 subsystem.
 *
 * This format is supported used by all platforms. For IA-64, older
 * applications using perfmon v2.0 MUST use the
 * perfmon_default_smpl.h
 */
#ifndef __PERFMON_DFL_SMPL_H__
#define __PERFMON_DFL_SMPL_H__ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <perfmon/perfmon.h>

#define PFM_DFL_SMPL_NAME	"default"

#ifdef PFMLIB_OLD_PFMV2
/*
 * UUID for compatibility with perfmon v2.2 (used by Cray)
 */
#define PFM_DFL_SMPL_UUID { \
       0xd1, 0x39, 0xb2, 0x9e, 0x62, 0xe8, 0x40, 0xe4,\
       0xb4, 0x02, 0x73, 0x07, 0x87, 0x92, 0xe9, 0x37 }
#endif

/*
 * format specific parameters (passed at context creation)
 */
typedef struct {
	uint64_t	buf_size;	/* size of the buffer in bytes */
	uint32_t	buf_flags;	/* buffer specific flags */
	uint32_t	res1;		/* for future use */
	uint64_t	reserved[6];	/* for future use */
} pfm_dfl_smpl_arg_t;

/*
 * This header is at the beginning of the sampling buffer returned to the user.
 * It is directly followed by the first record.
 */
typedef struct {
	uint64_t	hdr_count;              /* how many valid entries */
	uint64_t	hdr_cur_offs;           /* current offset from top of buffer */
	uint64_t	hdr_overflows;          /* #overflows for buffer */
	uint64_t	hdr_buf_size;           /* bytes in the buffer */
	uint64_t	hdr_min_buf_space;      /* minimal buffer size (internal use) */
	uint32_t	hdr_version;            /* smpl format version */
	uint32_t	hdr_buf_flags;          /* copy of buf_flags */
	uint64_t	hdr_reserved[10];       /* for future use */
} pfm_dfl_smpl_hdr_t;

/*
 * Entry header in the sampling buffer.  The header is directly followed
 * with the values of the PMD registers of interest saved in increasing 
 * index order: PMD4, PMD5, and so on. How many PMDs are present depends 
 * on how the session was programmed.
 *
 * In the case where multiple counters overflow at the same time, multiple
 * entries are written consecutively.
 *
 * last_reset_value member indicates the initial value of the overflowed PMD. 
 */
typedef struct {
        uint32_t	pid;                    /* thread id (for NPTL, this is gettid()) */
	uint16_t	ovfl_pmd;		/* index of pmd that overflowed for this sample */
	uint16_t	reserved;		/* for future use */
        uint64_t	last_reset_val;         /* initial value of overflowed PMD */
        uint64_t	ip;                     /* where did the overflow interrupt happened  */
        uint64_t	tstamp;                 /* overflow timetamp */
        uint16_t	cpu;                    /* cpu on which the overfow occured */
        uint16_t	set;                    /* event set active when overflow ocurred   */
        uint32_t	tgid;              	/* thread group id (for NPTL, this is getpid()) */
} pfm_dfl_smpl_entry_t;

#define PFM_DFL_SMPL_VERSION_MAJ	1U
#define PFM_DFL_SMPL_VERSION_MIN	0U
#define PFM_DFL_SMPL_VERSION	(((PFM_DFL_SMPL_VERSION_MAJ&0xffff)<<16)|(PFM_DFL_SMPL_VERSION_MIN & 0xffff))

#ifdef __cplusplus
};
#endif

#endif /* __PERFMON_DFL_SMPL_H__ */
