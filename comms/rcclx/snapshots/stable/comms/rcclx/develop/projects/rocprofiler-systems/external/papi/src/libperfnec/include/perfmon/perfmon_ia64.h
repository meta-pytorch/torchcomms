/*
 * Copyright (c) 2001-2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file should never be included directly, use
 * <perfmon/perfmon.h> instead.
 */

#ifndef _PERFMON_IA64_H_
#define _PERFMON_IA64_H_

#define PFM_ARCH_MAX_PMCS	(256+64) /* 256 HW 64 SW */
#define PFM_ARCH_MAX_PMDS	(256+64) /* 256 HW 64 SW */

/*
 * privilege level mask usage for ia-64:
 *
 * PFM_PLM0 = most privileged (kernel, hypervisor, ..)
 * PFM_PLM1 = privilege level 1
 * PFM_PLM2 = privilege level 2
 * PFM_PLM3 = least privileged (user level)
 */

/*
 * Itanium specific context flags
 */
#define PFM_ITA_FL_INSECURE 0x10000 /* force psr.sp=0 for non self-monitoring */

/*
 * Itanium specific event set flags
 */
#define PFM_ITA_SETFL_EXCL_INTR	0x10000	/* exclude interrupt triggered execution */
#define PFM_ITA_SETFL_INTR_ONLY	0x20000	/* include only interrupt triggered execution */
#define PFM_ITA_SETFL_IDLE_EXCL 0x40000 /* not stop monitoring in idle loop */

/*
 * compatibility for previous versions of the interface
 */
#include <perfmon/perfmon_compat.h>

#endif /* _PERFMON_IA64_H_ */
