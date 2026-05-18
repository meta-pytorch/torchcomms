/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    freebsd.c
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include <sys/types.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#include "papi.h"

#include "papi_internal.h"

#include "papi_lock.h"
#include "freebsd.h"
#include "papi_vector.h"

#include "map.h"

#include "freebsd-memory.h"
#include "x86_cpuid_info.h"

/* Global values referenced externally */
PAPI_os_info_t _papi_os_info;

/* Advance Declarations */
papi_vector_t _papi_freebsd_vector;
long long _papi_freebsd_get_real_cycles(void);
int _papi_freebsd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len);


/* For debugging */

static void show_counter(char *string, int id, char *name,
			 const char *function, char *file, int line) {

#if defined(DEBUG)
     pmc_value_t tmp_value;
     int ret = pmc_read (id, &tmp_value);
     
     fprintf(stderr,"%s\n",string);
     if (ret < 0) {
	fprintf (stderr, "DEBUG: Unable to read counter %s (ID: %08x) "
                         "on routine %s (file: %s, line: %d)\n", 
		         name, id, function,file,line);
     } else {
	fprintf (stderr, "DEBUG: Read counter %s (ID: %08x) - "
		         "value %llu on routine %s (file: %s, line: %d)\n", 
		         name, id, (long long unsigned int)tmp_value, 
		         function, file, line);
     }
#else
     (void) string; (void)name; 
     (void)id; (void)function; (void)file; (void)line;
#endif
}


static hwd_libpmc_context_t Context;


/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_freebsd_init_component.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the component level.
 */
int init_mdi(void)
{
	const struct pmc_cpuinfo *info;
   
	SUBDBG("Entering\n");

	/* Initialize PMC library */
	if (pmc_init() < 0)
		return PAPI_ESYS;
      
	if (pmc_cpuinfo (&info) != 0)
		return PAPI_ESYS;
   
	if (info != NULL)
	{
		/* Get CPU clock rate from HW.CLOCKRATE sysctl value, and
		   MODEL from HW.MODEL */
		int mib[5];
		size_t len;
		int hw_clockrate;
		char hw_model[PAPI_MAX_STR_LEN];
     
#if !defined(__i386__) && !defined(__amd64__)
		Context.use_rdtsc = FALSE;
#else
		/* Ok, I386s/AMD64s can use RDTSC. But be careful, if the cpufreq
		   module is loaded, then CPU frequency can vary and this method
		   does not work properly! We'll use use_rdtsc to know if this
		   method is available */
		len = 5; 
		Context.use_rdtsc = sysctlnametomib ("dev.cpufreq.0.%driver", mib, &len) == -1;
#endif

		len = 3;
		if (sysctlnametomib ("hw.clockrate", mib, &len) == -1)
			return PAPI_ESYS;
		len = sizeof(hw_clockrate);
		if (sysctl (mib, 2, &hw_clockrate, &len, NULL, 0) == -1)
			return PAPI_ESYS;

		len = 3;
		if (sysctlnametomib ("hw.model", mib, &len) == -1)
			return PAPI_ESYS;
		len = PAPI_MAX_STR_LEN;
		if (sysctl (mib, 2, &hw_model, &len, NULL, 0) == -1)
			return PAPI_ESYS;
		
		/*strcpy (_papi_hwi_system_info.hw_info.vendor_string, pmc_name_of_cputype(info->pm_cputype));*/
		sprintf (_papi_hwi_system_info.hw_info.vendor_string, "%s (TSC:%c)", pmc_name_of_cputype(info->pm_cputype), Context.use_rdtsc?'Y':'N');
		strcpy (_papi_hwi_system_info.hw_info.model_string, hw_model);
		_papi_hwi_system_info.hw_info.mhz = (float) hw_clockrate;
		_papi_hwi_system_info.hw_info.cpu_max_mhz = hw_clockrate;
		_papi_hwi_system_info.hw_info.cpu_min_mhz = hw_clockrate;
		_papi_hwi_system_info.hw_info.ncpu = info->pm_ncpu;
		_papi_hwi_system_info.hw_info.nnodes = 1;
		_papi_hwi_system_info.hw_info.totalcpus = info->pm_ncpu;
		/* Right now, PMC states that TSC is an additional counter. However
		   it's only available as a system-wide counter and this requires
		   root access */
		_papi_freebsd_vector.cmp_info.num_cntrs = info->pm_npmc - 1;

		if ( strstr(pmc_name_of_cputype(info->pm_cputype), "INTEL"))
		  _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_INTEL;
		else if ( strstr(pmc_name_of_cputype(info->pm_cputype), "AMD"))
		  _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_AMD;
		else
		  fprintf(stderr,"We didn't actually find a supported vendor...\n\n\n");
		}
		else
			return PAPI_ESYS;

	return 1;
}


int init_presets(int cidx)
{
	const struct pmc_cpuinfo *info;

	SUBDBG("Entering\n");

	if (pmc_cpuinfo (&info) != 0)
		return PAPI_ESYS;

	init_freebsd_libpmc_mappings();

	if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_P6") == 0)
		Context.CPUtype = CPU_P6;

	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PII") == 0)
		Context.CPUtype = CPU_P6_2;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PIII") == 0)
		Context.CPUtype = CPU_P6_3;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CL") == 0)
		Context.CPUtype = CPU_P6_C;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PM") == 0)
		Context.CPUtype = CPU_P6_M;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "AMD_K7") == 0)
		Context.CPUtype = CPU_K7;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "AMD_K8") == 0)
		Context.CPUtype = CPU_K8;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PIV") == 0)
		Context.CPUtype = CPU_P4;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_ATOM") == 0)
		Context.CPUtype = CPU_ATOM;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE") == 0)
		Context.CPUtype = CPU_CORE;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE2") == 0)
		Context.CPUtype = CPU_CORE2;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE2EXTREME") == 0)
		Context.CPUtype = CPU_CORE2EXTREME;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_COREI7") == 0)
		Context.CPUtype = CPU_COREI7;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_WESTMERE") == 0)
		Context.CPUtype = CPU_COREWESTMERE;
	else
		/* Unknown processor! */
		Context.CPUtype = CPU_UNKNOWN;


	_papi_freebsd_vector.cmp_info.num_native_events = freebsd_number_of_events (Context.CPUtype);
	_papi_freebsd_vector.cmp_info.attach = 0;

	_papi_load_preset_table((char *)pmc_name_of_cputype(info->pm_cputype),
				0,cidx);

	return 0;
}

/*
 * Component setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_freebsd_init_component(int cidx)
{
   (void)cidx;

   int retval;

   SUBDBG("Entering\n");

   /* Internal function, doesn't necessarily need to be a function */
   retval=init_presets(cidx);
   
   return retval;
}




/*
 * This is called whenever a thread is initialized
 */
int _papi_freebsd_init_thread(hwd_context_t *ctx)
{
    (void)ctx;
    SUBDBG("Entering\n");
    return PAPI_OK;
}

int _papi_freebsd_shutdown_thread(hwd_context_t *ctx)
{
  (void)ctx;
	SUBDBG("Entering\n");
	return PAPI_OK;
}

int _papi_freebsd_shutdown_component(void)
{
	SUBDBG("Entering\n");
	return PAPI_OK;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int _papi_freebsd_init_control_state(hwd_control_state_t *ptr)
{
	/* We will default to gather counters in USER|KERNEL mode */
	SUBDBG("Entering\n");
	ptr->hwc_domain = PAPI_DOM_USER|PAPI_DOM_KERNEL;
	ptr->pmcs = NULL;
	ptr->counters = NULL;
	ptr->n_counters = 0;
	return PAPI_OK;
}

int _papi_freebsd_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx)
{
	char name[1024];
	int i;
	int res;
	(void)ctx;

	SUBDBG("Entering\n");

	/* We're going to store which counters are being used in this EventSet.
	   As this ptr structure can be reused within many PAPI_add_event calls,
	   and domain can change we will reconstruct the table of counters
	   (ptr->counters) everytime where here.
	*/
	if (ptr->counters != NULL && ptr->n_counters > 0)
	{
		for (i = 0; i < ptr->n_counters; i++)
			if (ptr->counters[i] != NULL)
				free (ptr->counters[i]);
		free (ptr->counters);
	}
	if (ptr->pmcs != NULL)
		free (ptr->pmcs);
	if (ptr->values != NULL)
		free (ptr->values);
	if (ptr->caps != NULL)
		free (ptr->caps);

	ptr->n_counters = count;
	ptr->pmcs = (pmc_id_t*) malloc (sizeof(pmc_id_t)*count);
	ptr->caps = (uint32_t*) malloc (sizeof(uint32_t)*count);
	ptr->values = (pmc_value_t*) malloc (sizeof(pmc_value_t)*count);
	ptr->counters = (char **) malloc (sizeof(char*)*count);
	for (i = 0; i < count; i++)
		ptr->counters[i] = NULL;

	for (i = 0; i < count; i++)
	{
		res = _papi_freebsd_ntv_code_to_name (native[i].ni_event, name, sizeof(name));
		if (res != PAPI_OK)
			return res;

		native[i].ni_position = i;

		/* Domains can be applied to canonical events in libpmc (not "generic") */
		if (Context.CPUtype != CPU_UNKNOWN)
		{
			if (ptr->hwc_domain == (PAPI_DOM_USER|PAPI_DOM_KERNEL))
			{
				/* PMC defaults domain to OS & User. So simply copy the name of the counter */
				ptr->counters[i] = strdup (name);
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
			}
			else if (ptr->hwc_domain == PAPI_DOM_USER)
			{
				/* This is user-domain case. Just add unitmask=usr */
				ptr->counters[i] = malloc ((strlen(name)+strlen(",usr")+1)*sizeof(char));
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
				sprintf (ptr->counters[i], "%s,usr", name);
			}
			else /* if (ptr->hwc_domain == PAPI_DOM_KERNEL) */
			{
				/* This is the last case. Just add unitmask=os */
				ptr->counters[i] = malloc ((strlen(name)+strlen(",os")+1)*sizeof(char));
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
				sprintf (ptr->counters[i], "%s,os", name);
			}
		}
		else
		{
			/* PMC defaults domain to OS & User. So simply copy the name of the counter */
			ptr->counters[i] = strdup (name);
			if (ptr->counters[i] == NULL)
				return PAPI_ESYS;
		}
	}

	return PAPI_OK;
}

int _papi_freebsd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;
	(void)ctx;

	SUBDBG("Entering\n");

	for (i = 0; i < ctrl->n_counters; i++)
	{
		if ((ret = pmc_allocate (ctrl->counters[i], PMC_MODE_TC, 0, PMC_CPU_ANY, &(ctrl->pmcs[i]))) < 0)
		{
#if defined(DEBUG)
			/* This shouldn't happen, it's tested previously on _papi_freebsd_allocate_registers */
			fprintf (stderr, "DEBUG: %s FAILED to allocate '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
		if ((ret = pmc_capabilities (ctrl->pmcs[i],&(ctrl->caps[i]))) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to get capabilites for '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			ctrl->caps[i] = 0;
		}
#if defined(DEBUG)
		fprintf (stderr, "DEBUG: %s got counter '%s' is %swrittable! [%d of %d]\n", FUNC, ctrl->counters[i], (ctrl->caps[i]&PMC_CAP_WRITE)?"":"NOT", i+1, ctrl->n_counters);
#endif
		if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to start '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	}
	return PAPI_OK;
}

int _papi_freebsd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events, int flags)
{
	int i, ret;
	(void)ctx;
	(void)flags;

	SUBDBG("Entering\n");

	for (i = 0; i < ctrl->n_counters; i++)
		if ((ret = pmc_read (ctrl->pmcs[i], &(ctrl->values[i]))) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to read '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	*events = (long long *)ctrl->values;

#if defined(DEBUG)
	for (i = 0; i < ctrl->n_counters; i++)
		fprintf (stderr, "DEBUG: %s counter '%s' has value %lld\n", 
			 FUNC, ctrl->counters[i], (long long)ctrl->values[i]);
#endif
	return PAPI_OK;
}

int _papi_freebsd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;
	(void)ctx;

	SUBDBG("Entering\n");

	for (i = 0; i < ctrl->n_counters; i++)
	{
		if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
		if ((ret = pmc_release (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			/* This shouldn't happen, it's tested previously on _papi_freebsd_allocate_registers */
			fprintf (stderr, "DEBUG: %s FAILED to release '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	}
	return PAPI_OK;
}

int _papi_freebsd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;
	(void)ctx;

	SUBDBG("Entering\n");

	for (i = 0; i < ctrl->n_counters; i++)
	{
		/* Can we write on the counters? */
		if (ctrl->caps[i] & PMC_CAP_WRITE)
		{
			show_counter("DEBUG: _papi_freebsd_reset is about "
				     "to stop the counter i+1",
				     ctrl->pmcs[i],ctrl->counters[i],
				     __FUNCTION__,__FILE__,__LINE__);

			if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}

			show_counter(
				     "DEBUG: _papi_freebsd_reset is about "
				     "to write the counter i+1\n",
				     ctrl->pmcs[i],ctrl->counters[i],
				     __FUNCTION__,__FILE__,__LINE__);

			if ((ret = pmc_write (ctrl->pmcs[i], 0)) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to write '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}

			show_counter("DEBUG: _papi_freebsd_reset is about to "
				     "start the counter %i+1",
				     ctrl->pmcs[i],ctrl->counters[i],
				     __FUNCTION__,__FILE__,__LINE__);

			if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to start '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}

			show_counter("DEBUG: _papi_freebsd_reset after "
				     "starting the counter i+1",
				     ctrl->pmcs[i],ctrl->counters[i],
				     __FUNCTION__,__FILE__,__LINE__);

		}
		else
			return PAPI_ECMP;
	}
	return PAPI_OK;
}

int _papi_freebsd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *from)
{
	int i, ret;
	(void)ctx;

	SUBDBG("Entering\n");

	for (i = 0; i < ctrl->n_counters; i++)
	{
		/* Can we write on the counters? */
		if (ctrl->caps[i] & PMC_CAP_WRITE)
		{
			if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
			if ((ret = pmc_write (ctrl->pmcs[i], from[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to write '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
			if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
		}
		else
			return PAPI_ECMP;
	}
	return PAPI_OK;
}

/*
 * Overflow and profile functions 
 */
void _papi_freebsd_dispatch_timer(int signal, hwd_siginfo_t * info, void *context)
{
  (void)signal;
  (void)info;
  (void)context;
  /* Real function would call the function below with the proper args
   * _papi_hwi_dispatch_overflow_signal(...);
   */
  SUBDBG("Entering\n");
  return;
}

int _papi_freebsd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  (void)master;
  (void)ESI;
	SUBDBG("Entering\n");
  return PAPI_OK;
}

int _papi_freebsd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  (void)ESI;
  (void)EventIndex;
  (void)threshold;
	SUBDBG("Entering\n");
  return PAPI_OK;
}

int _papi_freebsd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
{
  (void)ESI;
  (void)EventIndex;
  (void)threashold;
	SUBDBG("Entering\n");
  return PAPI_OK;
}

/*
 * Functions for setting up various options
 */

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int _papi_freebsd_set_domain(hwd_control_state_t *cntrl, int domain) 
{
  int found = 0;

	SUBDBG("Entering\n");
	/* libpmc supports USER/KERNEL mode only when counters are native */
	if (Context.CPUtype != CPU_UNKNOWN)
	{
		if (domain & (PAPI_DOM_USER|PAPI_DOM_KERNEL))
		{
			cntrl->hwc_domain = domain & (PAPI_DOM_USER|PAPI_DOM_KERNEL);
			found = 1;
		}
		return found?PAPI_OK:PAPI_EINVAL;
	}
	else
		return PAPI_ECMP;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int _papi_freebsd_ctl (hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  (void)ctx;
	SUBDBG("Entering\n");
	switch (code)
	{
		case PAPI_DOMAIN:
		case PAPI_DEFDOM:
			/*return _papi_freebsd_set_domain(&option->domain.ESI->machdep, option->domain.domain);*/
			return _papi_freebsd_set_domain(option->domain.ESI->ctl_state, option->domain.domain);
		case PAPI_GRANUL:
		case PAPI_DEFGRN:
			return PAPI_ECMP;
		default:
			return PAPI_EINVAL;
   }
}


/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
long long _papi_freebsd_get_real_usec(void)
{
	/* Hey, I've seen somewhere a define called __x86_64__! Should I support it? */
#if !defined(__i386__) && !defined(__amd64__)
	/* This will surely work, but with low precision and high overhead */
	struct rusage res;

	SUBDBG("Entering\n");
	if ((getrusage(RUSAGE_SELF, &res) == -1))
		return PAPI_ESYS;
	return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
#else
	SUBDBG("Entering\n");
	if (Context.use_rdtsc)
	{
		return _papi_freebsd_get_real_cycles() / _papi_hwi_system_info.hw_info.cpu_max_mhz;
	}
	else
	{
		struct rusage res;
		if ((getrusage(RUSAGE_SELF, &res) == -1))
			return PAPI_ESYS;
		return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
	}
#endif
}


long long _papi_freebsd_get_real_cycles(void)
{
	/* Hey, I've seen somewhere a define called __x86_64__! Should I support it? */
#if !defined(__i386__) && !defined(__amd64__)
	SUBDBG("Entering\n");
	/* This will surely work, but with low precision and high overhead */
   return ((long long) _papi_freebsd_get_real_usec() * _papi_hwi_system_info.hw_info.cpu_max_mhz);
#else
	SUBDBG("Entering\n");
	if (Context.use_rdtsc)
	{
		long long cycles;
		__asm __volatile(".byte 0x0f, 0x31" : "=A" (cycles));
	  return cycles;
	}
	else
	{
		return ((long long) _papi_freebsd_get_real_usec() * _papi_hwi_system_info.hw_info.cpu_max_mhz);
	}
#endif
}



long long _papi_freebsd_get_virt_usec(void)
{
	struct rusage res;

	SUBDBG("Entering\n");

	if ((getrusage(RUSAGE_SELF, &res) == -1))
		return PAPI_ESYS;
	return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
}

/*
 * Native Event functions
 */


int _papi_freebsd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
     int res;
     char name[1024];
     unsigned int nextCode = 1 + *EventCode;

     SUBDBG("Entering\n");

     if (modifier==PAPI_ENUM_FIRST) {

       *EventCode=0;

	return PAPI_OK;
     }

     if (modifier==PAPI_ENUM_EVENTS) {

	res = _papi_freebsd_ntv_code_to_name(nextCode, name, sizeof(name));
	if (res != PAPI_OK) {
	      return res;
	} else {
	      *EventCode = nextCode;
	}
	return PAPI_OK;
     }

     return PAPI_ENOEVNT;

}

int _papi_freebsd_ntv_name_to_code(const char *name, unsigned int *event_code) {

   SUBDBG("Entering\n");

   int i;

   for(i = 0; i < _papi_freebsd_vector.cmp_info.num_native_events; i++) {
      if (strcmp (name, _papi_hwd_native_info[Context.CPUtype].info[i].name) == 0) {
	 *event_code = i;
	 return PAPI_OK;
      }
   }
   return PAPI_ENOEVNT;
}

int _papi_freebsd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, 
                                   int len)
{
    SUBDBG("Entering\n");

    int nidx;

    nidx = EventCode & PAPI_NATIVE_AND_MASK;
	
    if (nidx >= _papi_freebsd_vector.cmp_info.num_native_events) {
       return PAPI_ENOEVNT;
    }

    strncpy (ntv_name, 
	     _papi_hwd_native_info[Context.CPUtype].info[nidx].name, len);
    if (strlen(_papi_hwd_native_info[Context.CPUtype].info[nidx].name) > (size_t)len-1) {
		return PAPI_EBUF;
    }
    return PAPI_OK;
}

int _papi_freebsd_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
    SUBDBG("Entering\n");
    int nidx;

    nidx = EventCode & PAPI_NATIVE_AND_MASK;
    if (nidx >= _papi_freebsd_vector.cmp_info.num_native_events) {
       return PAPI_ENOEVNT;
    }

    strncpy (descr, _papi_hwd_native_info[Context.CPUtype].info[nidx].description, len);
    if (strlen(_papi_hwd_native_info[Context.CPUtype].info[nidx].description) > (size_t)len-1) {
       return PAPI_EBUF;
    }
    return PAPI_OK;
}


/* 
 * Counter Allocation Functions, only need to implement if
 *    the component needs smart counter allocation.
 */

/* Here we'll check if PMC can provide all the counters the user want */
int _papi_freebsd_allocate_registers (EventSetInfo_t *ESI) 
{
	char name[1024];
	int failed, allocated_counters, i, j, ret;
	pmc_id_t *pmcs;

	SUBDBG("Entering\n");

	failed = 0;
	pmcs = (pmc_id_t*) malloc(sizeof(pmc_id_t)*ESI->NativeCount);
	if (pmcs != NULL)
	{
		allocated_counters = 0;
		/* Check if we can allocate all the counters needed */
		for (i = 0; i < ESI->NativeCount; i++)
		{
			ret = _papi_freebsd_ntv_code_to_name (ESI->NativeInfoArray[i].ni_event, name, sizeof(name));
			if (ret != PAPI_OK)
				return ret;
			if ( (ret = pmc_allocate (name, PMC_MODE_TC, 0, PMC_CPU_ANY, &pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to allocate '%s' (%#08x) [%d of %d] ERROR = %d\n", FUNC, name, ESI->NativeInfoArray[i].ni_event, i+1, ESI->NativeCount, ret);
#endif
				failed = 1;
				break;
			}
			else
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s SUCCEEDED allocating '%s' (%#08x) [%d of %d]\n", FUNC, name, ESI->NativeInfoArray[i].ni_event, i+1, ESI->NativeCount);
#endif
				allocated_counters++;
			}
		}
		/* Free the counters */
		for (j = 0; j < allocated_counters; j++)
			pmc_release (pmcs[j]);
		free (pmcs);
	}
	else
		failed = 1;

	return failed?PAPI_ECNFLCT:PAPI_OK;
}

/*
 * Shared Library Information and other Information Functions
 */
int _papi_freebsd_update_shlib_info(papi_mdi_t *mdi){
	SUBDBG("Entering\n");
	(void)mdi;
  return PAPI_OK;
}



int
_papi_freebsd_detect_hypervisor(char *virtual_vendor_name) {

  int retval=0;

#if defined(__i386__)||defined(__x86_64__)
  retval=_x86_detect_hypervisor(virtual_vendor_name);
#else
  (void) virtual_vendor_name;
#endif
        
  return retval;
}



int
_papi_freebsd_get_system_info( papi_mdi_t *mdi ) {

  int retval;

  retval=_freebsd_get_memory_info(&mdi->hw_info, mdi->hw_info.model );

  /* Get virtualization info */
  mdi->hw_info.virtualized=_papi_freebsd_detect_hypervisor(mdi->hw_info.virtual_vendor_string);

  
  return PAPI_OK;

}

int 
_papi_hwi_init_os(void) {

   struct utsname uname_buffer;

   /* Internal function, doesn't necessarily need to be a function */
   init_mdi();

   uname(&uname_buffer);

   strncpy(_papi_os_info.name,uname_buffer.sysname,PAPI_MAX_STR_LEN);

   strncpy(_papi_os_info.version,uname_buffer.release,PAPI_MAX_STR_LEN);

   _papi_os_info.itimer_sig = PAPI_INT_MPX_SIGNAL;
   _papi_os_info.itimer_num = PAPI_INT_ITIMER;
   _papi_os_info.itimer_ns = PAPI_INT_MPX_DEF_US * 1000;	/* Not actually supported */
   _papi_os_info.itimer_res_ns = 1;

   _papi_freebsd_get_system_info(&_papi_hwi_system_info);

   return PAPI_OK;
}

papi_vector_t _papi_freebsd_vector = {
  .cmp_info = {
	/* default component information (unspecified values are initialized to 0) */
        .name = "FreeBSD",
	.description = "FreeBSD CPU counters",
	.default_domain = PAPI_DOM_USER,
	.available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
	.default_granularity = PAPI_GRN_THR,
	.available_granularities = PAPI_GRN_THR,

	.hardware_intr = 1,
	.kernel_multiplex = 1,
	.kernel_profile = 1,
	.num_mpx_cntrs = HWPMC_NUM_COUNTERS, /* ?? */
	.hardware_intr_sig = PAPI_INT_SIGNAL,

	/* component specific cmp_info initializations */
	.fast_real_timer = 1,
	.fast_virtual_timer = 0,
	.attach = 0,
	.attach_must_ptrace = 0,
  } ,
  .size = { 
	.context = sizeof( hwd_context_t ),
	.control_state = sizeof( hwd_control_state_t ),
	.reg_value = sizeof( hwd_register_t ),
	.reg_alloc = sizeof( hwd_reg_alloc_t )
  },

  .dispatch_timer	= _papi_freebsd_dispatch_timer,
  .start	= _papi_freebsd_start,
  .stop		= _papi_freebsd_stop,
  .read		= _papi_freebsd_read,
  .reset	= _papi_freebsd_reset,
  .write	= _papi_freebsd_write,
  .stop_profiling	= _papi_freebsd_stop_profiling,
  .init_component	= _papi_freebsd_init_component,
  .init_thread				= _papi_freebsd_init_thread,
  .init_control_state	= _papi_freebsd_init_control_state,
  .update_control_state	= _papi_freebsd_update_control_state,
  .ctl					= _papi_freebsd_ctl,
  .set_overflow		= _papi_freebsd_set_overflow,
  .set_profile		= _papi_freebsd_set_profile,
  .set_domain		= _papi_freebsd_set_domain,

  .ntv_enum_events	= _papi_freebsd_ntv_enum_events,
  .ntv_name_to_code	= _papi_freebsd_ntv_name_to_code,
  .ntv_code_to_name	= _papi_freebsd_ntv_code_to_name,
  .ntv_code_to_descr	= _papi_freebsd_ntv_code_to_descr,

  .allocate_registers	= _papi_freebsd_allocate_registers,

  .shutdown_thread	= _papi_freebsd_shutdown_thread,
  .shutdown_component	= _papi_freebsd_shutdown_component,
};

papi_os_vector_t _papi_os_vector = {
  .get_dmem_info	= _papi_freebsd_get_dmem_info,
  .get_real_cycles	= _papi_freebsd_get_real_cycles,
  .get_real_usec	= _papi_freebsd_get_real_usec,
  .get_virt_usec	= _papi_freebsd_get_virt_usec,
  .update_shlib_info	= _papi_freebsd_update_shlib_info,
  .get_system_info	= _papi_freebsd_get_system_info,
};
