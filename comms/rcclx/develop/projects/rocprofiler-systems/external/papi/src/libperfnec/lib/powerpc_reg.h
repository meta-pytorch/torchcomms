/*
 * These definitions were taken from the reg.h file which, until Linux
 * 2.6.18, resided in /usr/include/asm-ppc64.  Most of the unneeded
 * definitions have been removed, but there are still a few in this file
 * that are currently unused by libpfm.
 */

#ifndef _POWER_REG_H
#define _POWER_REG_H

#define __stringify_1(x)	#x
#define __stringify(x)		__stringify_1(x)

#define mfspr(rn)	({unsigned long rval; \
			asm volatile("mfspr %0," __stringify(rn) \
				: "=r" (rval)); rval;})

/* Special Purpose Registers (SPRNs)*/
#define SPRN_PVR	0x11F	/* Processor Version Register */

/* Performance monitor SPRs */
#define SPRN_MMCR0	795
#define   MMCR0_FC	0x80000000UL /* freeze counters */
#define   MMCR0_FCS	0x40000000UL /* freeze in supervisor state */
#define   MMCR0_KERNEL_DISABLE MMCR0_FCS
#define   MMCR0_FCP	0x20000000UL /* freeze in problem state */
#define   MMCR0_PROBLEM_DISABLE MMCR0_FCP
#define   MMCR0_FCM1	0x10000000UL /* freeze counters while MSR mark = 1 */
#define   MMCR0_FCM0	0x08000000UL /* freeze counters while MSR mark = 0 */
#define   MMCR0_PMXE	0x04000000UL /* performance monitor exception enable */
#define   MMCR0_FCECE	0x02000000UL /* freeze ctrs on enabled cond or event */
#define   MMCR0_TBEE	0x00400000UL /* time base exception enable */
#define   MMCR0_PMC1CE	0x00008000UL /* PMC1 count enable*/
#define   MMCR0_PMCjCE	0x00004000UL /* PMCj count enable*/
#define   MMCR0_TRIGGER	0x00002000UL /* TRIGGER enable */
#define   MMCR0_PMAO	0x00000080UL /* performance monitor alert has occurred, set to 0 after handling exception */
#define   MMCR0_SHRFC	0x00000040UL /* SHRre freeze conditions between threads */
#define   MMCR0_FC1_4   0x00000020UL /* freeze counters 1 - 4 on POWER5/5+ */
#define   MMCR0_FC5_6   0x00000010UL /* freeze counters 5 & 6 on POWER5/5+ */
#define   MMCR0_FCTI	0x00000008UL /* freeze counters in tags inactive mode */
#define   MMCR0_FCTA	0x00000004UL /* freeze counters in tags active mode */
#define   MMCR0_FCWAIT	0x00000002UL /* freeze counter in WAIT state */
#define   MMCR0_FCHV	0x00000001UL /* freeze conditions in hypervisor mode */
#define SPRN_MMCR1	798
#define SPRN_MMCRA	0x312
#define   MMCRA_SIHV	0x10000000UL /* state of MSR HV when SIAR set */
#define   MMCRA_SIPR	0x08000000UL /* state of MSR PR when SIAR set */
#define   MMCRA_SAMPLE_ENABLE 0x00000001UL /* enable sampling */
#define SPRN_PMC1	787
#define SPRN_PMC2	788
#define SPRN_PMC3	789
#define SPRN_PMC4	790
#define SPRN_PMC5	791
#define SPRN_PMC6	792
#define SPRN_PMC7	793
#define SPRN_PMC8	794
#define SPRN_SIAR	780
#define SPRN_SDAR	781

/* Processor Version Register (PVR) field extraction */

#define PVR_VER(pvr)	(((pvr) >>  16) & 0xFFFF)	/* Version field */
#define PVR_REV(pvr)	(((pvr) >>   0) & 0xFFFF)	/* Revison field */

#define __is_processor(pv)	(PVR_VER(mfspr(SPRN_PVR)) == (pv))

/* 64-bit processors */
/* XXX the prefix should be PVR_, we'll do a global sweep to fix it one day */
#define PV_NORTHSTAR	0x0033
#define PV_PULSAR	0x0034
#define PV_POWER4	0x0035
#define PV_ICESTAR	0x0036
#define PV_SSTAR	0x0037
#define PV_POWER4p	0x0038
#define PV_970		0x0039
#define PV_POWER5	0x003A
#define PV_POWER5p	0x003B
#define PV_970FX	0x003C
#define PV_POWER6	0x003E
#define PV_POWER7	0x003F
#define PV_630		0x0040
#define PV_630p		0x0041
#define PV_970MP	0x0044
#define PV_970GX	0x0045
#define PV_BE		0x0070

#endif /* _POWER_REG_H */
