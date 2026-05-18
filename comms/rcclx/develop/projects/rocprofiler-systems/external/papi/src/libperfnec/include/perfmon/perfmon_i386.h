/*
 * Copyright (c) 2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file should never be included directly, use
 * <perfmon/perfmon.h> instead.
 */

#ifndef _PERFMON_I386_H_
#define _PERFMON_I386_H_

/*
 *  Both i386 and x86-64 must have same limits to ensure ABI
 *  compatibility
 */
#define PFM_ARCH_MAX_PMCS	(256+64) /* 256 HW SW 64 */
#define PFM_ARCH_MAX_PMDS	(256+64) /* 256 HW SW 64 */

#endif /* _PERFMON_I386_H_ */
