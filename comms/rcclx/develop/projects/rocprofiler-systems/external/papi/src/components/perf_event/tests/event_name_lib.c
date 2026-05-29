#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"

char *get_offcore_event(char *event, int size) {

	const PAPI_hw_info_t *hwinfo;

	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		return NULL;
	}

	if (hwinfo->vendor == PAPI_VENDOR_INTEL) {

		if ( hwinfo->cpuid_family == 6) {
			switch(hwinfo->cpuid_model) {

			case 26:
			case 30:
			case 31: /* Nehalem */
			case 46: /* Nehalem EX */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:LOCAL_DRAM",size);
			return event;
			break;

			case 37:
			case 44: /* Westmere */
			case 47: /* Westmere EX */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:LOCAL_DRAM",size);
			return event;
			break;

			case 45: /* SandyBridge EP */
			case 42: /* SandyBridge */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 58: /* IvyBridge */
			case 62: /* Ivy Trail */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 60: /* Haswell */
			case 69:
			case 70:
			case 63: /* Haswell EP */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 61: /* Broadwell */
			case 71:
			case 86:
			case 79: /* Broadwell EP */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 78: /* Skylake */
			case 94:
			case 85: /* Skylake-X */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 142: /* Kabylake */
			case 158:
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			case 87: /* Knights Landing */
			strncpy(event,
			"OFFCORE_RESPONSE_0:DMND_DATA_RD:ANY_RESPONSE",size);
			return event;
			break;

			}
		}
		return NULL;
	}
	else if (hwinfo->vendor == PAPI_VENDOR_AMD) {
		return NULL;
	}

	return NULL;
}

char *get_instructions_event(char *event, int size) {

	const PAPI_hw_info_t *hwinfo;

	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		return NULL;
	}

	if (hwinfo->vendor == PAPI_VENDOR_INTEL) {

		if ( hwinfo->cpuid_family == 6) {
			strncpy(event,"INSTRUCTIONS_RETIRED",size);
			return event;
		}

		if ( hwinfo->cpuid_family == 15) {
			strncpy(event,"INSTR_RETIRED:NBOGUSNTAG",size);
			return event;
		}

		return NULL;
	}
	else if (hwinfo->vendor == PAPI_VENDOR_AMD) {
		strncpy(event,"RETIRED_INSTRUCTIONS",size);
		return event;
	}

	return NULL;
}

char *get_invalid_event_name(char *event, int size) {

	const PAPI_hw_info_t *hwinfo;

	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		return NULL;
	}

	if (hwinfo->vendor == PAPI_VENDOR_INTEL) {

		if ( hwinfo->cpuid_family == 6) {
			switch(hwinfo->cpuid_model) {

			case 78: /* Skylake */
			case 94:
			case 85: /* Skylake-X */
			strncpy(event,
			"DTLB_LOAD_MISSES:MISS_CAUSES_A_WALK,DTLB_STORE_MISSES:MISS_CAUSES_A_WALK:u=1",size);
			return event;
			break;

			}
		}
		return NULL;
	}
	else if (hwinfo->vendor == PAPI_VENDOR_AMD) {
		return NULL;
	}

	return NULL;
}

