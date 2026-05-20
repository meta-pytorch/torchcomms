/*
* File:    	native.c
* Mods: 	Maynard Johnson
*		maynardj@us.ibm.com
*/

/*
   This test defines an array of native event names, either at compile time
   or at run time (some x86 platforms). It then:
   - add the table of events to an event set;
   - starts counting
   - does a little work
   - stops counting;
   - reports the results.
*/

#include "papi_test.h"

static int EventSet = PAPI_NULL;
extern int TESTS_QUIET;				   /* Declared in test_utils.c */


#if (defined(PPC32))
   /* Select 4 events common to both ppc750 and ppc7450 */
static char *native_name[] = { "CPU_CLK", "FLOPS", "TOT_INS", "BR_MSP", NULL
};

#elif defined(_POWER4) || defined(_PPC970)
   /* arbitrarily code events from group 28: pm_fpu3 - Floating point events by unit */
static char *native_name[] =
	{ "PM_FPU0_FDIV", "PM_FPU1_FDIV", "PM_FPU0_FRSP_FCONV",
"PM_FPU1_FRSP_FCONV",
	"PM_FPU0_FMA", "PM_FPU1_FMA", "PM_INST_CMPL", "PM_CYC", NULL
};

#elif defined(_POWER5p)
/* arbitrarily code events from group 33: pm_fpustall - Floating Point Unit stalls */
static char *native_name[] =
	{ "PM_FPU_FULL_CYC", "PM_CMPLU_STALL_FDIV", "PM_CMPLU_STALL_FPU",
	"PM_RUN_INST_CMPL", "PM_RUN_CYC", NULL
};


#elif defined(_POWER5)
   /* arbitrarily code events from group 78: pm_fpu1 - Floating Point events */
static char *native_name[] =
	{ "PM_FPU_FDIV", "PM_FPU_FMA", "PM_FPU_FMOV_FEST", "PM_FPU_FEST",
	"PM_INST_CMPL", "PM_RUN_CYC", NULL
};

#elif defined(POWER3)
static char *native_name[] =
	{ "PM_IC_MISS", "PM_FPU1_CMPL", "PM_LD_MISS_L1", "PM_LD_CMPL",
	"PM_FPU0_CMPL", "PM_CYC", "PM_TLB_MISS", NULL
};

#elif defined(__ia64__)
#ifdef ITANIUM2
static char *native_name[] =
	{ "CPU_CYCLES", "L1I_READS", "L1D_READS_SET0", "IA64_INST_RETIRED", NULL
};
#else
static char *native_name[] =
	{ "DEPENDENCY_SCOREBOARD_CYCLE", "DEPENDENCY_ALL_CYCLE",
	"UNSTALLED_BACKEND_CYCLE", "MEMORY_CYCLE", NULL
};
#endif

#elif ((defined(linux) && (defined(__i386__) || (defined __x86_64__))) )
static char *p3_native_name[] = { "DATA_MEM_REFS", "DCU_LINES_IN", NULL };
static char *core_native_name[] = { "UnhltCore_Cycles", "Instr_Retired", NULL };
static char *k7_native_name[] =
	{ "TOT_CYC", "IC_MISSES", "DC_ACCESSES", "DC_MISSES", NULL };
//   static char *k8_native_name[] = { "FP_ADD_PIPE", "FP_MULT_PIPE", "FP_ST_PIPE", "FP_NONE_RET", NULL };
static char *k8_native_name[] =
	{ "DISPATCHED_FPU:OPS_ADD", "DISPATCHED_FPU:OPS_MULTIPLY",
"DISPATCHED_FPU:OPS_STORE", "CYCLES_NO_FPU_OPS_RETIRED", NULL };
static char *p4_native_name[] =
	{ "retired_mispred_branch_type:CONDITIONAL", "resource_stall:SBFULL",
	"tc_ms_xfer:CISC", "instr_retired:BOGUSNTAG:BOGUSTAG",
		"BSQ_cache_reference:RD_2ndL_HITS", NULL
};
static char **native_name = p3_native_name;

#elif defined(mips) && defined(sgi)
static char *native_name[] = { "Primary_instruction_cache_misses",
	"Primary_data_cache_misses", NULL
};
#elif defined(mips) && defined(linux)
static char *native_name[] = { "CYCLES", NULL };
#elif defined(sun) && defined(sparc)
static char *native_name[] = { "Cycle_cnt", "Instr_cnt", NULL };

#elif defined(_BGL)
static char *native_name[] =
	{ "BGL_UPC_PU0_PREF_STREAM_HIT", "BGL_PAPI_TIMEBASE",
"BGL_UPC_PU1_PREF_STREAM_HIT", NULL };

#elif defined(__bgp__)
static char *native_name[] =
	{ "PNE_BGP_PU0_JPIPE_LOGICAL_OPS", "PNE_BGP_PU0_JPIPE_LOGICAL_OPS",
"PNE_BGP_PU2_IPIPE_INSTRUCTIONS", NULL };

#else
#error "Architecture not supported in test file."
#endif


int
main( int argc, char **argv )
{
	int i, retval, native;
	const PAPI_hw_info_t *hwinfo;
	long long values[8];

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( ( hwinfo = PAPI_get_hardware_info(  ) ) == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", PAPI_EMISC );

	printf( "Architecture %s, %d\n", hwinfo->model_string, hwinfo->model );

#if ((defined(linux) && (defined(__i386__) || (defined __x86_64__))) )
	if ( !strncmp( hwinfo->model_string, "Intel Pentium 4", 15 ) ) {
		native_name = p4_native_name;
	} else if ( !strncmp( hwinfo->model_string, "AMD K7", 6 ) ) {
		native_name = k7_native_name;
	} else if ( !strncmp( hwinfo->model_string, "AMD K8", 6 ) ) {
		native_name = k8_native_name;
	} else if ( !strncmp( hwinfo->model_string, "Intel Core", 17 ) ||
				!strncmp( hwinfo->model_string, "Intel Core 2", 17 ) ) {
		native_name = core_native_name;
	}
#endif

	for ( i = 0; native_name[i] != NULL; i++ ) {
		retval = PAPI_event_name_to_code( native_name[i], &native );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
		printf( "Adding %s\n", native_name[i] );
		if ( ( retval = PAPI_add_event( EventSet, native ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_both( 1000 );

	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		for ( i = 0; native_name[i] != NULL; i++ ) {
			fprintf( stderr, "%-40s: ", native_name[i] );
			fprintf( stderr, LLDFMT, values[i] );
			fprintf( stderr, "\n" );
		}
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup", retval );
	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	test_pass( __FILE__, NULL, 0 );
	exit( 0 );
}
