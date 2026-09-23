/*
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This header file contains obsolete user-level perfmon interface
 * definitions for the Itanium Processor Family architecture.
 *
 * please use replacements as indicated below whenever possible.
 */

#ifndef _PERFMON_COMPAT_H_
#define _PERFMON_COMPAT_H_

#ifndef __ia64__
#error "you should not include this file on non Itanium platforms"
#endif

/*
 * old perfmon2 interface for backward compatibility.
 * Do not use in portable applications.
 */
extern int perfmonctl(int fd, int cmd, void *arg, int narg);

typedef unsigned char pfm_uuid_t[16];	/* custom sampling buffer identifier type */

/*
 * obsolete perfmon comamnds supported on all CPU models
 */
#define PFM_WRITE_PMCS		0x01 
#define PFM_WRITE_PMDS		0x02 
#define PFM_READ_PMDS		0x03 
#define PFM_STOP		0x04
#define PFM_START		0x05
#define PFM_ENABLE		0x06 /* obsolete */
#define PFM_DISABLE		0x07 /* obsolete */
#define PFM_CREATE_CONTEXT	0x08
#define PFM_DESTROY_CONTEXT	0x09 /* obsolete use close() */
#define PFM_RESTART		0x0a
#define PFM_PROTECT_CONTEXT	0x0b /* obsolete */
#define PFM_GET_FEATURES	0x0c /* obsolete: use /proc/sys/kernel/perfmon */
#define PFM_DEBUG		0x0d /* obsolete: use /proc/sys/kernel/perfmon/debug */
#define PFM_UNPROTECT_CONTEXT	0x0e /* obsolete */
#define PFM_GET_PMC_RESET_VAL	0x0f /* obsolete: use /proc/perfmon_mappings */
#define PFM_LOAD_CONTEXT	0x10
#define PFM_UNLOAD_CONTEXT	0x11

/*
 * PMU model specific commands (may not be supported on all PMU models)
 */
#define PFM_WRITE_IBRS		0x20 /* obsolete: use PFM_WRITE_PMCS[256-263] */
#define PFM_WRITE_DBRS		0x21 /* obsolete: use PFM_WRITE_PMCS[264-271] */

/*
 * argument to PFM_CREATE_CONTEXT
 */
typedef struct {
	pfm_uuid_t     ctx_smpl_buf_id;	 /* which buffer format to use (if needed) */
	unsigned long  ctx_flags;	 /* noblock/block */
	unsigned int   ctx_reserved1;	 /* for future use */
	int	       ctx_fd;		 /* return arg: unique identification for context */
	void	       *ctx_smpl_vaddr;	 /* return arg: virtual address of sampling buffer, is used */
	unsigned long  ctx_reserved3[11];/* for future use */
} pfarg_context_t;


/*
 * argument structure for PFM_WRITE_PMCS/PFM_WRITE_PMDS/PFM_WRITE_PMDS
 */
typedef struct {
	unsigned int	reg_num;	   /* which register */
	unsigned short	reg_set;	   /* event set for this register */
	unsigned short	reg_reserved1;	   /* for future use */

	unsigned long	reg_value;	   /* initial pmc/pmd value */
	unsigned long	reg_flags;	   /* input: pmc/pmd flags, return: reg error */

	unsigned long	reg_long_reset;	   /* reset after buffer overflow notification */
	unsigned long	reg_short_reset;   /* reset after counter overflow */

	unsigned long	reg_reset_pmds[4]; /* which other counters to reset on overflow */
	unsigned long	reg_random_seed;   /* seed value when randomization is used */
	unsigned long	reg_random_mask;   /* bitmask used to limit random value */
	unsigned long   reg_last_reset_val;/* return: PMD last reset value */

	unsigned long	reg_smpl_pmds[4];  	/* which pmds are accessed when PMC overflows */
	unsigned long	reg_smpl_eventid;  	/* opaque sampling event identifier */
	unsigned long   reg_ovfl_switch_cnt;	/* how many overflow before switch for next set */

	unsigned long   reg_reserved2[2];   /* for future use */
} pfarg_reg_t;

/*
 * argument to PFM_WRITE_IBRS/PFM_WRITE_DBRS
 */
typedef struct {
	unsigned int	dbreg_num;		/* which debug register */
	unsigned short	dbreg_set;		/* event set for this register */
	unsigned short	dbreg_reserved1;	/* for future use */
	unsigned long	dbreg_value;		/* value for debug register */
	unsigned long	dbreg_flags;		/* return: dbreg error */
	unsigned long	dbreg_reserved2[1];	/* for future use */
} pfarg_dbreg_t;

/*
 * argument to PFM_GET_FEATURES
 */
typedef struct {
	unsigned int	ft_version;	/* perfmon: major [16-31], minor [0-15] */
	unsigned int	ft_reserved;	/* reserved for future use */
	unsigned long	reserved[4];	/* for future use */
} pfarg_features_t;

typedef struct {
	int		msg_type;		/* generic message header */
	int		msg_ctx_fd;		/* generic message header */
	unsigned long	msg_ovfl_pmds[4];	/* which PMDs overflowed */
	unsigned short  msg_active_set;		/* active set at the time of overflow */
	unsigned short  msg_reserved1;		/* for future use */
	unsigned int    msg_reserved2;		/* for future use */
	unsigned long	msg_tstamp;		/* for perf tuning/debug */
} pfm_ovfl_msg_t;

typedef struct {
	int		msg_type;		/* generic message header */
	int		msg_ctx_fd;		/* generic message header */
	unsigned long	msg_tstamp;		/* for perf tuning */
} pfm_end_msg_t;

typedef struct {
	int		msg_type;		/* type of the message */
	int		msg_ctx_fd;		/* unique identifier for the context */
	unsigned long	msg_tstamp;		/* for perf tuning */
} pfm_gen_msg_t;

typedef union {
	int type;
	pfm_ovfl_msg_t	pfm_ovfl_msg;
	pfm_end_msg_t	pfm_end_msg;
	pfm_gen_msg_t	pfm_gen_msg;
} pfm_msg_t;

/*
 * PMD/PMC return flags in case of error (ignored on input)
 *
 * Those flags are used on output and must be checked in case EINVAL is returned
 * by a command accepting a vector of values and each has a flag field, such as
 * pfarg_pmc_t or pfarg_pmd_t.
 */
#define PFM_REG_RETFL_NOTAVAIL	(1<<31) /* set if register is implemented but not available */
#define PFM_REG_RETFL_EINVAL	(1<<30) /* set if register entry is invalid */
#define PFM_REG_RETFL_MASK	(PFM_REG_RETFL_NOTAVAIL|PFM_REG_RETFL_EINVAL)

#define PFM_REG_HAS_ERROR(flag)	(((flag) & PFM_REG_RETFL_MASK) != 0)

#endif /* _PERFMON_COMPAT_H_ */
