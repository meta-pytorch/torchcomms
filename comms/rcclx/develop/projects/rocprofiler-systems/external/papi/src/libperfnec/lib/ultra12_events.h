static pme_sparc_entry_t ultra12_pe[] = {
	/* These two must always be first.  */
	{	.pme_name = "Cycle_cnt",
		.pme_desc = "Accumulated cycles",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x0,
	},
	{	.pme_name = "Instr_cnt",
		.pme_desc = "Number of instructions completed",
		.pme_ctrl = PME_CTRL_S0 | PME_CTRL_S1,
		.pme_val = 0x1,
	},
	{
		.pme_name = "Dispatch0_IC_miss",
		.pme_desc = "I-buffer is empty from I-Cache miss",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0x2,
	},

	/* PIC0 events for UltraSPARC-I/II/IIi/IIe */
	{
		.pme_name = "Dispatch0_storeBuf",
		.pme_desc = "Store buffer can not hold additional stores",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0x3,
	},
	{
		.pme_name = "IC_ref",
		.pme_desc = "I-cache references",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0x8,
	},
	{
		.pme_name = "DC_rd",
		.pme_desc = "D-cache read references (including accesses that subsequently trap)",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0x9,
	},
	{
		.pme_name = "DC_wr",
		.pme_desc = "D-cache write references (including accesses that subsequently trap)",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xa,
	},
	{
		.pme_name = "Load_use",
		.pme_desc = "An instruction in the execute stage depends on an earlier load result that is not yet available",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xb,
	},
	{
		.pme_name = "EC_ref",
		.pme_desc = "Total E-cache references",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xc,
	},
	{
		.pme_name = "EC_write_hit_RDO",
		.pme_desc = "E-cache hits that do a read for ownership UPA transaction",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xd,
	},
	{
		.pme_name = "EC_snoop_inv",
		.pme_desc = "E-cache invalidates from the following UPA transactions: S_INV_REQ, S_CPI_REQ",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xe,
	},
	{
		.pme_name = "EC_rd_hit",
		.pme_desc = "E-cache read hits from D-cache misses",
		.pme_ctrl = PME_CTRL_S0,
		.pme_val = 0xf,
	},

	/* PIC1 events for UltraSPARC-I/II/IIi/IIe */
	{
		.pme_name = "Dispatch0_mispred",
		.pme_desc = "I-buffer is empty from Branch misprediction",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0x2,
	},
	{
		.pme_name = "Dispatch0_FP_use",
		.pme_desc = "First instruction in the group depends on an earlier floating point result that is not yet available",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0x3,
	},
	{
		.pme_name = "IC_hit",
		.pme_desc = "I-cache hits",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0x8,
	},
	{
		.pme_name = "DC_rd_hit",
		.pme_desc = "D-cache read hits",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0x9,
	},
	{
		.pme_name = "DC_wr_hit",
		.pme_desc = "D-cache write hits",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xa,
	},
	{
		.pme_name = "Load_use_RAW",
		.pme_desc = "There is a load use in the execute stage and there is a read-after-write hazard on the oldest outstanding load",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xb,
	},
	{
		.pme_name = "EC_hit",
		.pme_desc = "Total E-cache hits",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xc,
	},
	{
		.pme_name = "EC_wb",
		.pme_desc = "E-cache misses that do writebacks",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xd,
	},
	{
		.pme_name = "EC_snoop_cb",
		.pme_desc = "E-cache snoop copy-backs from the following UPA transactions: S_CPB_REQ, S_CPI_REQ, S_CPD_REQ, S_CPB_MIS_REQ",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xe,
	},
	{
		.pme_name = "EC_ic_hit",
		.pme_desc = "E-cache read hits from I-cache misses",
		.pme_ctrl = PME_CTRL_S1,
		.pme_val = 0xf,
	},
};
#define PME_ULTRA12_EVENT_COUNT	   (sizeof(ultra12_pe)/sizeof(pme_sparc_entry_t))
