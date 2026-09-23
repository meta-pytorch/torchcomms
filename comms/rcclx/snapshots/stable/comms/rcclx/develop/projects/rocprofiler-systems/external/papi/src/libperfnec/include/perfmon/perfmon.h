/*
 * This file contains the user level interface description for
 * the perfmon3.x interface on Linux.
 *
 * It also includes perfmon2.x interface definitions.
 *
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 */
#ifndef __PERFMON_H__
#define __PERFMON_H__

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __x86_64__
#include <perfmon/perfmon_x86_64.h>
#endif

#define PFM_MAX_PMCS	PFM_ARCH_MAX_PMCS
#define PFM_MAX_PMDS	PFM_ARCH_MAX_PMDS

#ifndef SWIG
/*
 * number of element for each type of bitvector
 */
#define PFM_BPL		(sizeof(uint64_t)<<3)
#define PFM_BVSIZE(x)   (((x)+PFM_BPL-1) / PFM_BPL)
#define PFM_PMD_BV      PFM_BVSIZE(PFM_MAX_PMDS)
#define PFM_PMC_BV      PFM_BVSIZE(PFM_MAX_PMCS)
#endif

/*
 * special data type for syscall error value used to help
 * with Python support and in particular for SWIG. By using
 * a specific type we can detect syscalls and trap errors
 * in one SWIG statement as opposed to having to keep track of
 * each syscall individually. Programs can use 'int' safely for
 * the return value.
 */
typedef int os_err_t;			/* error if -1 */

/*
 * passed to pfm_create
 * contains list of available register upon return
 */

#define PFM_ARCH_MAX_PMCS 32
#define PFM_ARCH_MAX_PMDS 32
typedef struct {
	uint64_t	sif_avail_pmcs[PFM_PMC_BV]; /* out: available PMCs */
	uint64_t	sif_avail_pmds[PFM_PMD_BV]; /* out: available PMDs */
	uint64_t	sif_reserved[4];
} pfarg_sinfo_t;

//os_err_t pfm_create(int flags, pfarg_sinfo_t *sif,
//		      char *smpl_name, void *smpl_arg, size_t arg_size);
extern os_err_t pfm_create(int flags, pfarg_sinfo_t *sif, ...);


/*
 * pfm_create flags:
 * bits[00-15]: generic flags
 * bits[16-31]: arch-specific flags (see perfmon_const.h)
 */
#define PFM_FL_NOTIFY_BLOCK	0x01 /* block task on user notifications */
#define PFM_FL_SYSTEM_WIDE	0x02 /* create a system wide context */
#define PFM_FL_SMPL_FMT		0x04 /* session uses sampling format */
#define PFM_FL_OVFL_NO_MSG	0x80 /* no overflow msgs */

/*
 * PMC and PMD generic (simplified) register description
 */
typedef struct {
	uint16_t reg_num;	/* which register */
	uint16_t reg_set;	/* which event set */
	uint32_t reg_flags;	/* REGFL flags */
	uint64_t reg_value;	/* 64-bit value */
} pfarg_pmr_t;

/*
 * pfarg_pmr_t flags:
 * bit[00-15] : generic flags
 * bit[16-31] : arch-specific flags
 *
 * PFM_REGFL_NO_EMUL64: must be set on the PMC controlling the PMD
 */
#define PFM_REGFL_OVFL_NOTIFY	0x1	/* PMD: send notification on event */
#define PFM_REGFL_RANDOM	0x2	/* PMD: randomize value after event */
#define PFM_REGFL_NO_EMUL64	0x4	/* PMC: no 64-bit emulation */

/* 
 * PMD extended description
 * to be used with pfm_writeand pfm_read
 * must be used with type = PFM_RW_PMD_ATTR
 */
typedef struct {
	uint16_t reg_num;		/* which register */
	uint16_t reg_set;		/* which event set */
	uint32_t reg_flags;		/* REGFL flags */
	uint64_t reg_value;		/* 64-bit value */
	uint64_t reg_long_reset;	/* write: value to reload after notification */
	uint64_t reg_short_reset;	/* write: reset after counter overflow */
	uint64_t reg_random_mask; 	/* write: bitmask used to limit random value */
	uint64_t reg_smpl_pmds[PFM_PMD_BV];  /* write: record in sample */
	uint64_t reg_reset_pmds[PFM_PMD_BV]; /* write: reset on overflow */
	uint64_t reg_ovfl_swcnt;	/* write: # overflows before switch */
	uint64_t reg_smpl_eventid;	/* write: opaque event identifier */
	uint64_t reg_last_value;	/* read: PMD last reset value */
	uint64_t reg_reserved[8];	/* for future use */
} pfarg_pmd_attr_t;


/*
 * pfm_write, pfm_read type:
 */
#define PFM_RW_PMD	1 /* simplified PMD (pfarg_pmr_t) */
#define PFM_RW_PMC	2 /* PMC registers (pfarg_pmr_t) */
#define PFM_RW_PMD_ATTR	3 /* extended PMD (pfarg_pmd_attr) */

/*
 * pfm_attach special target for detach
 */
#define PFM_NO_TARGET	-1 /* no target, detach */


/*
 * pfm_set_state state:
 */
#define PFM_ST_START		0x1 /* start monitoring */
#define PFM_ST_STOP		0x2 /* stop monitoring */
#define PFM_ST_RESTART		0x3 /* resume after notify */

#ifndef PFMLIB_OLD_PFMV2
typedef struct {
	uint16_t	set_id;		  /* which set */
	uint16_t	set_reserved1;	  /* for future use */
	uint32_t    	set_flags; 	  /* SETFL flags */
	uint64_t	set_timeout;	  /* requested/effective switch timeout in nsecs */
	uint64_t	reserved[6];	  /* for future use */
} pfarg_set_desc_t;

typedef struct {
        uint16_t	set_id;             /* which set */
        uint16_t	set_reserved1;      /* for future use */
        uint32_t	set_reserved2;	    /* for future use */
        uint64_t 	set_ovfl_pmds[PFM_PMD_BV]; /* out: last ovfl PMDs */
        uint64_t	set_runs;           /* out: #times set was active */
        uint64_t	set_timeout;        /* out: leftover switch timeout (nsecs) */
	uint64_t	set_duration;	    /* out: time set was active (nsecs) */
        uint64_t	set_reserved3[4];   /* for future use */
} pfarg_set_info_t;
#endif

/*
 * pfm_set_desc_t flags:
 */
#define PFM_SETFL_OVFL_SWITCH	0x01 /* enable switch on overflow (subject to individual switch_cnt */
#define PFM_SETFL_TIME_SWITCH	0x02 /* switch set on timeout */

#ifndef PFMLIB_OLD_PFMV2
typedef struct {
	uint32_t 	msg_type;		/* PFM_MSG_OVFL */
	uint32_t 	msg_ovfl_pid;		/* process id */
	uint16_t 	msg_active_set;		/* active set at the time of overflow */
	uint16_t 	msg_ovfl_cpu;		/* cpu on which the overflow occurred */
	uint32_t	msg_ovfl_tid;		/* thread id */
	uint64_t	msg_ovfl_ip;		/* instruction pointer where overflow interrupt happened */
	uint64_t	msg_ovfl_pmds[PFM_PMD_BV];/* which PMDs overflowed */
} pfarg_ovfl_msg_t;

extern os_err_t pfm_write(int fd, int flags, int type, void *reg, size_t n);
extern os_err_t pfm_read(int fd, int flags, int type, void *reg, size_t n);
extern os_err_t pfm_set_state(int fd, int flags, int state);
extern os_err_t pfm_create_sets(int fd, int flags, pfarg_set_desc_t *s, size_t sz);
extern os_err_t pfm_getinfo_sets(int fd, int flags, pfarg_set_info_t *s, size_t sz);
extern os_err_t pfm_attach(int fd, int flags, int target);

#endif

#include "perfmon_v2.h"

typedef union {
	uint32_t		type;
        pfarg_ovfl_msg_t	pfm_ovfl_msg;
} pfarg_msg_t;

#define PFM_MSG_OVFL	1	/* an overflow happened */
#define PFM_MSG_END	2	/* thread to which context was attached ended */

#define PFM_VERSION_MAJOR(x)	 (((x)>>16) & 0xffff)
#define PFM_VERSION_MINOR(x)	 ((x) & 0xffff)

#ifdef __cplusplus
};
#endif

#endif /* _PERFMON_H */
