#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"

char *get_uncore_event(char *event, int size) {

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
			strncpy(event,"nhm_unc::UNC_CLK_UNHALTED",size);
			return event;
			break;

			case 37:
			case 44: /* Westmere */
			case 47: /* Westmere EX */
			strncpy(event,"wsm_unc::UNC_CLK_UNHALTED",size);
			return event;
			break;

			case 42: /* SandyBridge */
			strncpy(event,"snb_unc_cbo0::UNC_CLOCKTICKS",size);
			return event;
			break;

			case 58: /* IvyBridge */
			strncpy(event,"ivb_unc_cbo0::UNC_CLOCKTICKS",size);
			return event;
			break;

			case 62: /* Ivy Trail */
			case 45: /* SandyBridge EP */
			strncpy(event,"snbep_unc_imc0::UNC_M_CLOCKTICKS",size);
			return event;
			break;

			case 60:
			case 70:
			case 69: /* Haswell: note libpfm4 has no haswell unc support */
			return NULL;
			break;

			case 63: /*haswell EP*/
			strncpy(event,"hswep_unc_cbo0::UNC_C_CLOCKTICKS",size);
			return event;
			break;

			case 61:
			case 71:
			case 86: /* Broadwell: note libpfm4 has no broadwell unc support */
			return NULL;
			break;

			case 79: /* Broadwell-EP */
			strncpy(event,"bdx_unc_cbo0::UNC_C_CLOCKTICKS",size);
			return event;
			break;

			case 78:
			case 94: /* Skylake: note libpfm4 has no skylake unc support */
			return NULL;
			break;

			case 85: /* Skylake-X */
				/* note libpfm4 has no skylake-x unc support */
			return NULL;
			break;

			case 87: /*Knights Landing*/
			strncpy(event,"knl_unc_imc0::UNC_M_D_CLOCKTICKS",size);
			return event;
			break;
			}
		}
		return NULL;
	}
	else if (hwinfo->vendor == PAPI_VENDOR_AMD) {
		if ( hwinfo->cpuid_family == 21) {
			/* For kernel 3.9 at least */
			strncpy(event,"DRAM_ACCESSES:ALL",size);
			return event;
		}
		return NULL;
	}

	return NULL;
}


char *get_uncore_cbox_event(char *event_name, char *uncore_base, int size) {

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
			/* No CBOX event? */
			return NULL;
			break;

			case 37:
			case 44: /* Westmere */
			case 47: /* Westmere EX */
			/* No CBOX event? */
			return NULL;
			break;

			case 42: /* SandyBridge */
			strncpy(event_name,"UNC_CBO_CACHE_LOOKUP:STATE_I:ANY_FILTER",size);
			strncpy(uncore_base,"snb_unc_cbo",size);
			return event_name;
			break;

			case 58: /* IvyBridge */
			strncpy(event_name,"UNC_CBO_CACHE_LOOKUP:STATE_I:ANY_FILTER",size);
			strncpy(uncore_base,"ivb_unc_cbo",BUFSIZ);
			return event_name;
			break;

			case 62: /* Ivy Trail */
			case 45: /* SandyBridge EP */
			strncpy(event_name,"UNC_C_TOR_OCCUPANCY:ALL",size);
			strncpy(uncore_base,"snbep_unc_cbo",size);
			return event_name;
			break;

			case 60:
			case 70:
			case 69: /* Haswell: note libpfm4 has no haswell unc support */
			return NULL;
			break;

			case 63: /*haswell EP*/
			strncpy(event_name,"UNC_C_COUNTER0_OCCUPANCY",size);
			strncpy(uncore_base,"hswep_unc_cbo",size);
			return event_name;
			break;

			case 61:
			case 71:
			case 86: /* Broadwell: note libpfm4 has no broadwell unc support */
			return NULL;
			break;

			case 79: /* Broadwell-EP */
			strncpy(event_name,"UNC_C_COUNTER0_OCCUPANCY",size);
			strncpy(uncore_base,"bdx_unc_cbo",size);
			return event_name;
			break;

			case 78:
			case 94: /* Skylake: note libpfm4 has no skylake unc support */
			return NULL;
			break;

			case 85: /* Skylake-X */
				/* note libpfm4 has no skylake-x unc support */
			return NULL;
			break;

			case 87: /*Knights Landing*/
			strncpy(event_name,"UNC_M_D_CLOCKTICKS",size);
			strncpy(uncore_base,"knl_unc_imc",size);
			return event_name;
			break;
			}
		}
		return NULL;
	}

	return NULL;
}
