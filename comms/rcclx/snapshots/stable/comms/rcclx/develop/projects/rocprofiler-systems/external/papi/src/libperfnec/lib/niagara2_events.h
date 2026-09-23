static pme_sparc_mask_entry_t niagara2_pe[] = {
	/* PIC0 Niagara-2 events */
	{	.pme_name = "All_strands_idle",
		.pme_desc = "Cycles when no strand can be picked for the physical core on which the monitoring strand resides.",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x0,
		.pme_masks = {
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored2",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored3",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored4",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored5",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored6",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored7",
				.mask_desc = "Ignored",
			},
		},
	},
	{	.pme_name = "Instr_cnt",
		.pme_desc = "Number of instructions completed",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x2,
		.pme_masks = {
			{
				.mask_name = "branches",
				.mask_desc = "Completed branches",
			},
			{
				.mask_name = "taken_branches",
				.mask_desc = "Taken branches, which are always mispredicted",
			},
			{
				.mask_name = "FGU_arith",
				.mask_desc = "All FADD, FSUB, FCMP, convert, FMUL, FDIV, FNEG, FABS, FSQRT, FMOV, FPADD, FPSUB, FPACK, FEXPAND, FPMERGE, FMUL8, FMULD8, FALIGNDATA, BSHUFFLE, FZERO, FONE, FSRC, FNOT1, FNOT2, FOR, FNOR, FAND, FNAND, FXOR, FXNOR, FORNOT1, FORNOT2, FANDNOT1, FANDNOT2, PDIST, SIAM",

			},
			{
				.mask_name = "Loads",
				.mask_desc = "Load instructions",
			},
			{
				.mask_name = "Stores",
				.mask_desc = "Stores instructions",
			},
			{
				.mask_name = "SW_count",
				.mask_desc = "Software count 'sethi %hi(fc00), %g0' instructions",
			},
			{
				.mask_name = "other",
				.mask_desc = "Instructions not covered by other mask bits",
			},
			{
				.mask_name = "atomics",
				.mask_desc = "Atomics are LDSTUB/A, CASA/XA, SWAP/A",
			},
		},
	},
	{
		.pme_name = "cache",
		.pme_desc = "Cache events",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x3,
		.pme_masks = {
			{
				.mask_name = "IC_miss",
				.mask_desc = "I-cache misses. This counts only primary instruction cache misses, and does not count duplicate instruction cache misses.4 Also, only 'true' misses are counted. If a thread encounters an I$ miss, but the thread is redirected (due to a branch misprediction or trap, for example) before the line returns from L2 and is loaded into the I$, then the miss is not counted.",
			},
			{
				.mask_name = "DC_miss",
				.mask_desc = "D-cache misses.  This counts both primary and duplicate data cache misses.",
			},
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "L2IC_miss",
				.mask_desc = "L2 cache instruction misses",
			},
			{
				.mask_name = "L2LD_miss",
				.mask_desc = "L2 cache load misses.  Block loads are treated as one L2 miss event. In reality, each individual load can hit or miss in the L2 since the block load is not atomic.",
			},
			{
				.mask_name = "ignored2",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored3",
				.mask_desc = "Ignored",
			},
		},
	},
	{
		.pme_name = "TLB",
		.pme_desc = "TLB events",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x4,
		.pme_masks = {
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ITLB_L2ref",
				.mask_desc = "ITLB references to L2. For each ITLB miss with hardware tablewalk enabled, count each access the ITLB hardware tablewalk makes to L2.",
			},
			{
				.mask_name = "DTLB_L2ref",
				.mask_desc = "DTLB references to L2. For each DTLB miss with hardware tablewalk enabled, count each access the DTLB hardware tablewalk makes to L2.",
			},
			{
				.mask_name = "ITLB_L2miss",
				.mask_desc = "For each ITLB miss with hardware tablewalk enabled, count each access the ITLB hardware tablewalk makes to L2 which misses in L2.  Note: Depending upon the hardware table walk configuration, each ITLB miss may issue from 1 to 4 requests to L2 to search TSBs.",
			},
			{
				.mask_name = "DTLB_L2miss",
				.mask_desc = "For each DTLB miss with hardware tablewalk enabled, count each access the DTLB hardware tablewalk makes to L2 which misses in L2.  Note: Depending upon the hardware table walk configuration, each DTLB miss may issue from 1 to 4 requests to L2 to search TSBs.",

			},
			{
				.mask_name = "ignored2",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored3",
				.mask_desc = "Ignored",
			},
		},
	},
	{
		.pme_name = "mem",
		.pme_desc = "Memory operations",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x5,
		.pme_masks = {
			{
				.mask_name = "stream_load",
				.mask_desc = "Stream Unit load operations to L2",
			},
			{
				.mask_name = "stream_store",
				.mask_desc = "Stream Unit store operations to L2",
			},
			{
				.mask_name = "cpu_load",
				.mask_desc = "CPU loads to L2",
			},
			{
				.mask_name = "cpu_ifetch",
				.mask_desc = "CPU instruction fetches to L2",
			},
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "cpu_store",
				.mask_desc = "CPU stores to L2",
			},
			{
				.mask_name = "mmu_load",
				.mask_desc = "MMU loads to L2",
			},
		},
	},
	{
		.pme_name = "spu_ops",
		.pme_desc = "Stream Unit operations.  User, supervisor, and hypervisor counting must all be enabled to properly count these events.",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x6,
		.pme_masks = {
			{
				.mask_name = "DES",
				.mask_desc = "Increment for each CWQ or ASI operation that uses DES/3DES unit",
			},
			{
				.mask_name = "AES",
				.mask_desc = "Increment for each CWQ or ASI operation that uses AES unit",
			},
			{
				.mask_name = "RC4",
				.mask_desc = "Increment for each CWQ or ASI operation that uses RC4 unit",
			},
			{
				.mask_name = "HASH",
				.mask_desc = "Increment for each CWQ or ASI operation that uses MD5/SHA-1/SHA-256 unit",
			},
			{
				.mask_name = "MA",
				.mask_desc = "Increment for each CWQ or ASI modular arithmetic operation",
			},
			{
				.mask_name = "CSUM",
				.mask_desc = "Increment for each iSCSI CRC or TCP/IP checksum operation",
			},
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
		},
	},
	{
		.pme_name = "spu_busy",
		.pme_desc = "Stream Unit busy cycles.  User, supervisor, and hypervisor counting must all be enabled to properly count these events.",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x07,
		.pme_masks = {
			{
				.mask_name = "DES",
				.mask_desc = "Cycles the DES/3DES unit is busy",
			},
			{
				.mask_name = "AES",
				.mask_desc = "Cycles the AES unit is busy",
			},
			{
				.mask_name = "RC4",
				.mask_desc = "Cycles the RC4 unit is busy",
			},
			{
				.mask_name = "HASH",
				.mask_desc = "Cycles the MD5/SHA-1/SHA-256 unit is busy",
			},
			{
				.mask_name = "MA",
				.mask_desc = "Cycles the modular arithmetic unit is busy",
			},
			{
				.mask_name = "CSUM",
				.mask_desc = "Cycles the CRC/MPA/checksum unit is busy",
			},
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
		},
	},
	{
		.pme_name = "tlb_miss",
		.pme_desc = "TLB misses",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0xb,
		.pme_masks = {
			{
				.mask_name = "ignored0",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored1",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ITLB",
				.mask_desc = "I-TLB misses",
			},
			{
				.mask_name = "DTLB",
				.mask_desc = "D-TLB misses",
			},
			{
				.mask_name = "ignored2",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored3",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored4",
				.mask_desc = "Ignored",
			},
			{
				.mask_name = "ignored5",
				.mask_desc = "Ignored",
			},
		},
	},
};
#define PME_NIAGARA2_EVENT_COUNT	   (sizeof(niagara2_pe)/sizeof(pme_sparc_mask_entry_t))
