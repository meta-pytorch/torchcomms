/*
 * Copyright (c) 2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file should never be included directly, use
 * <perfmon/perfmon.h> instead.
 */

#ifndef _PERFMON_CRAY_H_
#define _PERFMON_CRAY_H_

#define PFM_ARCH_MAX_PMCS	(12+8)	/* 12 HW SW 8 */
#define PFM_ARCH_MAX_PMDS	(512+8)	/* 512 HW SW 8 */

/*
 * Cray specific register flags
 */
#define PFM_CRAY_REGFL_SMP_SCOPE 0x10000 /* PMD: shared state event counter */

#endif /* _PERFMON_CRAY_H_ */
