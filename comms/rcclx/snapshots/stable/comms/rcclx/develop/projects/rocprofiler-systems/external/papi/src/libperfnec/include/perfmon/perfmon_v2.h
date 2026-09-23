/*
 * This file contains the user level interface description for
 * the perfmon-2.x interface on Linux.
 *
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 */
#ifndef __PERFMON_V2_H__
#define __PERFMON_V2_H__

#ifndef __PERFMON_H__
#error "this file should never be included directly, use perfmon.h instead"
#endif

/*
 * argument to v2.3 and onward pfm_create_context()
 */
typedef struct {
	uint32_t	ctx_flags;	   /* noblock/block/syswide */
	uint32_t	ctx_reserved1;	   /* for future use */
	uint64_t	ctx_reserved3[7];  /* for future use */
} pfarg_ctx_t;

/*
 * argument for pfm_write_pmcs()
 */
typedef struct {
	uint16_t reg_num;	   			/* which register */
	uint16_t reg_set;	   			/* event set for this register */
	uint32_t reg_flags;	   			/* REGFL flags */
	uint64_t reg_value;	   			/* pmc value */
	uint64_t reg_reserved2[4];			/* for future use */
} pfarg_pmc_t;

/*
 * argument pfm_write_pmds() and pfm_read_pmds()
 */
typedef struct {
	uint16_t reg_num;	   	/* which register */
	uint16_t reg_set;	   	/* event set for this register */
	uint32_t reg_flags;	   	/* REGFL flags */
	uint64_t reg_value;	   	/* initial pmc/pmd value */
	uint64_t reg_long_reset;	/* reset after buffer overflow notification */
	uint64_t reg_short_reset;   	/* reset after counter overflow */
	uint64_t reg_last_reset_val;	/* return: PMD last reset value */
	uint64_t reg_ovfl_switch_cnt;	/* how many overflow before switch for next set */
	uint64_t reg_reset_pmds[PFM_PMD_BV]; /* which other PMDS to reset on overflow */
	uint64_t reg_smpl_pmds[PFM_PMD_BV];  /* which other PMDS to record when the associated PMD overflows */
	uint64_t reg_smpl_eventid;  	/* opaque sampling event identifier */
	uint64_t reg_random_mask; 	/* bitmask used to limit random value */
	uint32_t reg_random_seed;   	/* seed for randomization (DEPRECATED) */
	uint32_t reg_reserved2[7];	/* for future use */
} pfarg_pmd_t;

/*
 * optional argument to pfm_start(), pass NULL if no arg needed
 */
typedef struct {
	uint16_t start_set;		/* event set to start with */
	uint16_t start_reserved1;	/* for future use */
	uint32_t start_reserved2;	/* for future use */
	uint64_t reserved3[3];		/* for future use */
} pfarg_start_t;

/*
 * argument to pfm_load_context()
 */
typedef struct {
	uint32_t	load_pid;          /* thread or CPU to attach to */
	uint16_t        load_set;          /* set to load first */
	uint16_t        load_reserved1;    /* for future use */
	uint64_t        load_reserved2[3]; /* for future use */
} pfarg_load_t;

#ifndef PFMLIB_OLD_PFMV2
typedef struct {
	uint16_t	set_id;		  /* which set */
	uint16_t	set_reserved1;	  /* for future use */
	uint32_t    	set_flags; 	  /* SETFL flags */
	uint64_t	set_timeout;	  /* requested/effective switch timeout in nsecs */
	uint64_t	reserved[6];	  /* for future use */
} pfarg_setdesc_t;

typedef struct {
        uint16_t	set_id;             /* which set */
        uint16_t	set_reserved1;      /* for future use */
        uint32_t	set_flags;          /* for future use */
        uint64_t 	set_ovfl_pmds[PFM_PMD_BV]; /* out: last ovfl PMDs */
        uint64_t	set_runs;           /* out: #times set was active */
        uint64_t	set_timeout;        /* out: leftover switch timeout (nsecs) */
	uint64_t	set_act_duration;	    /* out: time set was active (nsecs) */
	uint64_t	set_avail_pmcs[PFM_PMC_BV]; /* out: available PMCs */
	uint64_t	set_avail_pmds[PFM_PMD_BV]; /* out: available PMDs */
        uint64_t	set_reserved3[6];   /* for future use */
} pfarg_setinfo_t;
#endif


#ifdef PFMLIB_OLD_PFMV2

/*
 * argument to pfm_create_evtsets()/pfm_delete_evtsets()
 */
typedef struct {
 	uint16_t	set_id;		  /* which set */
	uint16_t	set_id_next;	  /* next set to go to (must use PFM_SETFL_EXPL_NEXT) */
 	uint32_t    	set_flags; 	  /* SETFL flags */
 	uint64_t	set_timeout;	  /* requested switch timeout in nsecs */
	uint64_t	set_mmap_offset;  /* cookie to pass as mmap offset to access 64-bit virtual PMD */
	uint64_t	reserved[5];	  /* for future use */
} pfarg_setdesc_t;

/*
 * argument to pfm_getinfo_evtsets()
 */
typedef struct {
	uint16_t	set_id;             /* which set */
	uint16_t	set_id_next;        /* output: next set to go to (must use PFM_SETFL_EXPL_NEXT) */
	uint32_t	set_flags;          /* output: SETFL flags */
	uint64_t 	set_ovfl_pmds[PFM_PMD_BV]; /* output: last ovfl PMDs which triggered a switch from set */
	uint64_t	set_runs;           /* output: number of times the set was active */
	uint64_t	set_timeout;        /* output:effective/leftover switch timeout in nsecs */
	uint64_t	set_act_duration;   /* number of cycles set was active (syswide only) */
	uint64_t	set_mmap_offset;    /* cookie to pass as mmap offset to access 64-bit virtual PMD */
	uint64_t	set_avail_pmcs[PFM_PMC_BV];
	uint64_t	set_avail_pmds[PFM_PMD_BV];
	uint64_t	reserved[4];        /* for future use */
} pfarg_setinfo_t;

#ifdef __crayx2
#define PFM_MAX_HW_PMDS 512
#else
#define PFM_MAX_HW_PMDS 256
#endif
#define PFM_HW_PMD_BV   PFM_BVSIZE(PFM_MAX_HW_PMDS)

typedef struct {
	uint32_t 	msg_type;		/* PFM_MSG_OVFL */
	uint32_t 	msg_ovfl_pid;		/* process id */
	uint64_t	msg_ovfl_pmds[PFM_HW_PMD_BV];/* which PMDs overflowed */
	uint16_t 	msg_active_set;		/* active set at the time of overflow */
	uint16_t 	msg_ovfl_cpu;		/* cpu on which the overflow occurred */
	uint32_t	msg_ovfl_tid;		/* thread id */
	uint64_t	msg_ovfl_ip;		/* instruction pointer where overflow interrupt happened */
} pfarg_ovfl_msg_t;

#endif /* PFMLIB_OLD_PFMV2 */

extern os_err_t pfm_create_context(pfarg_ctx_t *ctx, char *smpl_name,
                                    void *smpl_arg, size_t smpl_size);
extern os_err_t pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count);
extern os_err_t pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count);
extern os_err_t pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count);
extern os_err_t pfm_load_context(int fd, pfarg_load_t *load);
extern os_err_t pfm_start(int fd, pfarg_start_t *start);
extern os_err_t pfm_stop(int fd);
extern os_err_t pfm_restart(int fd);
extern os_err_t pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count);
extern os_err_t pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count);
extern os_err_t pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count);
extern os_err_t pfm_unload_context(int fd);

#endif /* _PERFMON_V2_H */
