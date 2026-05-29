/****************************
THIS IS OPEN SOURCE CODE

Part of the PAPI software library. Copyright (c) 2005 - 2017,
Innovative Computing Laboratory, Dept of Electrical Engineering &
Computer Science University of Tennessee, Knoxville, TN.

The open source software license conforms to the 2-clause BSD License
template.

****************************/

/**
 * @file    linux-nvml.c
 * @author  Kiran Kumar Kasichayanula
 *          kkasicha@utk.edu
 * @author  James Ralph
 *          ralph@eecs.utk.edu
 * @ingroup papi_components
 *
 * @brief This is an NVML component, it demos the component interface
 *  and implements a number of counters from the Nvidia Management
 *  Library. Please refer to NVML documentation for details about
 *  nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature. Power is
 *  reported in mW and temperature in Celcius.  The counter
 *  descriptions should contain the units that the measurement
 *  returns.
 */
#include <dlfcn.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "linux-nvml.h"

#include "nvml.h"

void (*_dl_non_dynamic_init)(void) __attribute__((weak));

/*****  CHANGE PROTOTYPES TO DECLARE CUDA AND NVML LIBRARY SYMBOLS AS WEAK  *****
 *  This is done so that a version of PAPI built with the NVML component can    *
 *  be installed on a system which does not have the NVML libraries installed.  *
 *                                                                              *
 *  If this is done without these prototypes, then all papi services on the     *
 *  system without the NVML libraries installed will fail.  The PAPI libraries  *
 *  contain references to the NVML libraries which are not installed.  The      *
 *  load of PAPI commands fails because the NVML library references can not be  *
 *  resolved.                                                                   *
 *                                                                              *
 *  This also defines pointers to the NVML library functions that we call.      *
 *  These function pointers will be resolved with dlopen/dlsym calls at         *
 *  component initialization time.  The component then calls the NVML library   *
 *  functions through these function pointers.                                  *
 ********************************************************************************/
#undef DECLDIR
#define DECLDIR __attribute__((weak))
nvmlReturn_t DECLDIR nvmlDeviceGetClockInfo(nvmlDevice_t, nvmlClockType_t, unsigned int *);
const char*  DECLDIR nvmlErrorString(nvmlReturn_t);
nvmlReturn_t DECLDIR nvmlDeviceGetDetailedEccErrors(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetFanSpeed(nvmlDevice_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo(nvmlDevice_t, nvmlMemory_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetPerformanceState(nvmlDevice_t, nvmlPstates_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetTemperature(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetTotalEccErrors(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, unsigned long long *);
nvmlReturn_t DECLDIR nvmlDeviceGetUtilizationRates(nvmlDevice_t, nvmlUtilization_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByIndex(unsigned int, nvmlDevice_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo(nvmlDevice_t, nvmlPciInfo_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetName(nvmlDevice_t, char *, unsigned int);
nvmlReturn_t DECLDIR nvmlDeviceGetInforomVersion(nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int);
nvmlReturn_t DECLDIR nvmlDeviceGetEccMode(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
nvmlReturn_t DECLDIR nvmlInit(void);
nvmlReturn_t DECLDIR nvmlDeviceGetCount(unsigned int *);
nvmlReturn_t DECLDIR nvmlShutdown(void);
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit);
nvmlReturn_t DECLDIR nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int  limit);
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit);

static nvmlReturn_t (*nvmlDeviceGetClockInfoPtr)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
static char* (*nvmlErrorStringPtr)(nvmlReturn_t);
static nvmlReturn_t (*nvmlDeviceGetDetailedEccErrorsPtr)(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *);
static nvmlReturn_t (*nvmlDeviceGetFanSpeedPtr)(nvmlDevice_t, unsigned int *);
static nvmlReturn_t (*nvmlDeviceGetMemoryInfoPtr)(nvmlDevice_t, nvmlMemory_t *);
static nvmlReturn_t (*nvmlDeviceGetPerformanceStatePtr)(nvmlDevice_t, nvmlPstates_t *);
static nvmlReturn_t (*nvmlDeviceGetPowerUsagePtr)(nvmlDevice_t, unsigned int *);
static nvmlReturn_t (*nvmlDeviceGetTemperaturePtr)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
static nvmlReturn_t (*nvmlDeviceGetTotalEccErrorsPtr)(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, unsigned long long *);
static nvmlReturn_t (*nvmlDeviceGetUtilizationRatesPtr)(nvmlDevice_t, nvmlUtilization_t *);
static nvmlReturn_t (*nvmlDeviceGetHandleByIndexPtr)(unsigned int, nvmlDevice_t *);
static nvmlReturn_t (*nvmlDeviceGetPciInfoPtr)(nvmlDevice_t, nvmlPciInfo_t *);
static nvmlReturn_t (*nvmlDeviceGetNamePtr)(nvmlDevice_t, char *, unsigned int);
static nvmlReturn_t (*nvmlDeviceGetInforomVersionPtr)(nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int);
static nvmlReturn_t (*nvmlDeviceGetEccModePtr)(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
static nvmlReturn_t (*nvmlInitPtr)(void);
static nvmlReturn_t (*nvmlDeviceGetCountPtr)(unsigned int *);
static nvmlReturn_t (*nvmlShutdownPtr)(void);
static nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitPtr)(nvmlDevice_t device, unsigned int* limit);
static nvmlReturn_t (*nvmlDeviceSetPowerManagementLimitPtr)(nvmlDevice_t device, unsigned int  limit);
static nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitConstraintsPtr)(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit);

// file handles used to access NVML libraries with dlopen
static void* dl3 = NULL;

static char nvml_main[]=PAPI_NVML_MAIN;

static int linkCudaLibraries();

/* Declare our vector in advance */
papi_vector_t _nvml_vector;

/* upto 25 events per card how many cards per system should we allow for?! */
#define NVML_MAX_COUNTERS 100

/** Holds control flags.  Usually there's one of these per event-set.
 *    Usually this is out-of band configuration of the hardware
 */
typedef struct nvml_control_state {
    int num_events;
    int which_counter[NVML_MAX_COUNTERS];
    long long counter[NVML_MAX_COUNTERS];   /**< Copy of counts, holds results when stopped */
} nvml_control_state_t;

/** Holds per-thread information */
typedef struct nvml_context {
    nvml_control_state_t state;
} nvml_context_t;

/** This table contains the native events */
static nvml_native_event_entry_t *nvml_native_table = NULL;
static                       int *nvml_dev_id_table = NULL;

/** Number of devices detected at component_init time */
static int device_count = 0;

/** number of events in the table*/
static int num_events = 0;

static nvmlDevice_t* devices = NULL;
static int* features = NULL;
static unsigned int *power_management_initial_limit = NULL;
static unsigned int *power_management_limit_constraint_min = NULL;
static unsigned int *power_management_limit_constraint_max = NULL;

unsigned long long
getClockSpeed(nvmlDevice_t dev, nvmlClockType_t which_one)
{
    unsigned int ret = 0;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetClockInfoPtr)(dev, which_one, &ret);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }

    return (unsigned long long)ret;
}

unsigned long long
getEccLocalErrors(nvmlDevice_t dev, nvmlEccBitType_t bits, int which_one)
{
    nvmlEccErrorCounts_t counts;

    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetDetailedEccErrorsPtr)(dev, bits, NVML_VOLATILE_ECC , &counts);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    switch (which_one) {
    case LOCAL_ECC_REGFILE:
        return counts.registerFile;
    case LOCAL_ECC_L1:
        return counts.l1Cache;
    case LOCAL_ECC_L2:
        return counts.l2Cache;
    case LOCAL_ECC_MEM:
        return counts.deviceMemory;
    default:
        ;
    }
    return (unsigned long long) - 1;
}

unsigned long long
getFanSpeed(nvmlDevice_t dev)
{
    unsigned int ret = 0;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetFanSpeedPtr)(dev, &ret);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    return (unsigned long long)ret;
}

unsigned long long
getMaxClockSpeed(nvmlDevice_t dev, nvmlClockType_t which_one)
{
    unsigned int ret = 0;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetClockInfoPtr)(dev, which_one, &ret);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    return (unsigned long long) ret;
}

unsigned long long
getMemoryInfo(nvmlDevice_t dev, int which_one)
{
    nvmlMemory_t meminfo;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetMemoryInfoPtr)(dev, &meminfo);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }

    switch (which_one) {
    case MEMINFO_TOTAL_MEMORY:
        return meminfo.total;
    case MEMINFO_UNALLOCED:
        return meminfo.free;
    case MEMINFO_ALLOCED:
        return meminfo.used;
    default:
        ;
    }
    return (unsigned long long) - 1;
}

unsigned long long
getPState(nvmlDevice_t dev)
{
    unsigned int ret = 0;
    nvmlPstates_t state = NVML_PSTATE_15;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetPerformanceStatePtr)(dev, &state);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    switch (state) {
    case NVML_PSTATE_15:
        ret++;
        // fall through
    case NVML_PSTATE_14:
        ret++;
        // fall through
    case NVML_PSTATE_13:
        ret++;
        // fall through
    case NVML_PSTATE_12:
        ret++;
        // fall through
    case NVML_PSTATE_11:
        ret++;
        // fall through
    case NVML_PSTATE_10:
        ret++;
        // fall through
    case NVML_PSTATE_9:
        ret++;
        // fall through
    case NVML_PSTATE_8:
        ret++;
        // fall through
    case NVML_PSTATE_7:
        ret++;
        // fall through
    case NVML_PSTATE_6:
        ret++;
        // fall through
    case NVML_PSTATE_5:
        ret++;
        // fall through
    case NVML_PSTATE_4:
        ret++;
        // fall through
    case NVML_PSTATE_3:
        ret++;
        // fall through
    case NVML_PSTATE_2:
        ret++;
        // fall through
    case NVML_PSTATE_1:
        ret++;
        // fall through
    case NVML_PSTATE_0:
        break;
        // fall through
    case NVML_PSTATE_UNKNOWN:
    default:
        /* This should never happen?
         * The API docs just state Unknown performance state... */
        return (unsigned long long) - 1;
    }
    return (unsigned long long)ret;
}

unsigned long long
getPowerUsage(nvmlDevice_t dev)
{
    unsigned int power;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetPowerUsagePtr)(dev, &power);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    return (unsigned long long) power;
}

unsigned long long
getTemperature(nvmlDevice_t dev)
{
    unsigned int ret = 0;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetTemperaturePtr)(dev, NVML_TEMPERATURE_GPU, &ret);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    return (unsigned long long)ret;
}

unsigned long long
getTotalEccErrors(nvmlDevice_t dev, nvmlEccBitType_t bits)
{
    unsigned long long counts = 0;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetTotalEccErrorsPtr)(dev, bits, NVML_VOLATILE_ECC , &counts);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }
    return counts;
}

/*  0 => gpu util
    1 => memory util
 */
unsigned long long
getUtilization(nvmlDevice_t dev, int which_one)
{
    nvmlUtilization_t util;
    nvmlReturn_t bad;
    bad = (*nvmlDeviceGetUtilizationRatesPtr)(dev, &util);

    if (NVML_SUCCESS != bad) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(bad));
        return (unsigned long long) - 1;
    }

    switch (which_one) {
    case GPU_UTILIZATION:
        return (unsigned long long) util.gpu;
    case MEMORY_UTILIZATION:
        return (unsigned long long) util.memory;
    default:
        ;
    }

    return (unsigned long long) - 1;
}

unsigned long long getPowerManagementLimit(nvmlDevice_t dev)
{
    unsigned int limit;
    nvmlReturn_t rv;
    rv = (*nvmlDeviceGetPowerManagementLimitPtr)(dev, &limit);
    if (NVML_SUCCESS != rv) {
        SUBDBG("something went wrong %s\n", (*nvmlErrorStringPtr)(rv));
        return (unsigned long long) 0;
    }
    return (unsigned long long) limit;
}

static int _papi_nvml_init_private(void);

/*
 * Check for the initialization step and does it if needed
 */
static int
_nvml_check_n_initialize(papi_vector_t *vector)
{
  if (!vector->cmp_info.initialized)
      return _papi_nvml_init_private();
  return PAPI_OK;
}

#define DO_SOME_CHECKING(vectorp) do {           \
  int err = _nvml_check_n_initialize(vectorp);   \
  if (PAPI_OK != err) return err;                \
} while(0)

static void
nvml_hardware_reset()
{
    /* nvmlDeviceSet* and nvmlDeviceClear* calls require root/admin access, so while
     * possible to implement a reset on the ECC counters, we pass */
    /*
       for ( i=0; i < device_count; i++ )
       nvmlDeviceClearEccErrorCounts( device[i], NVML_VOLATILE_ECC );
    */
    int i;
    nvmlReturn_t ret;
    unsigned int templimit = 0;
    for (i = 0; i < device_count; i++) {
        if (HAS_FEATURE(features[i], FEATURE_POWER_MANAGEMENT)) {
            // if power management is available
            if (power_management_initial_limit[i] != 0) {
                ret = (*nvmlDeviceGetPowerManagementLimitPtr)(devices[i], &templimit);
                if ((ret == NVML_SUCCESS) && (templimit != power_management_initial_limit[i])) {
                    SUBDBG("Reset power_management_limit on device %d to initial value of %d \n", i, power_management_initial_limit[i]);
                    // if power is not at its initial value
                    // reset to initial value
                    ret = (*nvmlDeviceSetPowerManagementLimitPtr)(devices[i], power_management_initial_limit[i]);
                    if (ret != NVML_SUCCESS)
                        SUBDBG("Unable to reset the NVML power management limit on device %i to %ull (return code %d) \n", i, power_management_initial_limit[i] , ret);
                }
            }
        }
    }
}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
static int
nvml_hardware_read(long long *value, int which_one)
//, nvml_context_t *ctx)
{
    nvml_native_event_entry_t *entry;
    nvmlDevice_t handle;
    int cudaIdx = -1;

    entry = &nvml_native_table[which_one];
    *value = (long long) - 1;
    /* replace entry->resources with the current cuda_device->nvml device */
    cudaIdx = nvml_dev_id_table[which_one];

    if (cudaIdx < 0 || cudaIdx > device_count)
        return PAPI_EINVAL;

    /* Make sure the device we are running on has the requested event */
    if (!HAS_FEATURE(features[cudaIdx] , entry->type))
        return PAPI_EINVAL;

    handle = devices[cudaIdx];

    switch (entry->type) {
    case FEATURE_CLOCK_INFO:
        *value =  getClockSpeed(handle, (nvmlClockType_t)entry->options.clock);
        break;
    case FEATURE_ECC_LOCAL_ERRORS:
        *value = getEccLocalErrors(handle,
                                   (nvmlEccBitType_t)entry->options.ecc_opts.bits,
                                   (int)entry->options.ecc_opts.which_one);
        break;
    case FEATURE_FAN_SPEED:
        *value = getFanSpeed(handle);
        break;
    case FEATURE_MAX_CLOCK:
        *value = getMaxClockSpeed(handle,
                                  (nvmlClockType_t)entry->options.clock);
        break;
    case FEATURE_MEMORY_INFO:
        *value = getMemoryInfo(handle,
                               (int)entry->options.which_one);
        break;
    case FEATURE_PERF_STATES:
        *value = getPState(handle);
        break;
    case FEATURE_POWER:
        *value = getPowerUsage(handle);
        break;
    case FEATURE_TEMP:
        *value = getTemperature(handle);
        break;
    case FEATURE_ECC_TOTAL_ERRORS:
        *value = getTotalEccErrors(handle,
                                   (nvmlEccBitType_t)entry->options.ecc_opts.bits);
        break;
    case FEATURE_UTILIZATION:
        *value = getUtilization(handle,
                                (int)entry->options.which_one);
        break;
    case FEATURE_POWER_MANAGEMENT:
        *value = getPowerManagementLimit(handle);
        break;

    case FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN:
        *value = power_management_limit_constraint_min[cudaIdx];
        break;

    case FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX:
        *value = power_management_limit_constraint_max[cudaIdx];
        break;

    default:
        return PAPI_EINVAL;
    }
    if (*value == (long long)(unsigned long long) - 1)
        return PAPI_EINVAL;

    return PAPI_OK;
}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
static int nvml_hardware_write(long long *value, int which_one)
{
    nvml_native_event_entry_t *entry;
    nvmlDevice_t handle;
    int cudaIdx = -1;
    nvmlReturn_t nvret;

    entry = &nvml_native_table[which_one];
    /* replace entry->resources with the current cuda_device->nvml device */
    cudaIdx = nvml_dev_id_table[which_one];

    if (cudaIdx < 0 || cudaIdx > device_count)
        return PAPI_EINVAL;

    /* Make sure the device we are running on has the requested event */
    if (!HAS_FEATURE(features[cudaIdx] , entry->type))
        return PAPI_EINVAL;

    handle = devices[cudaIdx];

    switch (entry->type) {
    case FEATURE_POWER_MANAGEMENT: {
        unsigned int setToPower = (unsigned int) * value;
        if (setToPower < power_management_limit_constraint_min[cudaIdx]) {
            SUBDBG("Error: Desired power %u mW < minimum %u mW on device %d\n", setToPower, power_management_limit_constraint_min[cudaIdx], cudaIdx);
            return PAPI_EINVAL;
        }
        if (setToPower > power_management_limit_constraint_max[cudaIdx]) {
            SUBDBG("Error: Desired power %u mW > maximum %u mW on device %d\n", setToPower, power_management_limit_constraint_max[cudaIdx], cudaIdx);
            return PAPI_EINVAL;
        }
        if ((nvret = (*nvmlDeviceSetPowerManagementLimitPtr)(handle, setToPower)) != NVML_SUCCESS) {
            SUBDBG("Error: %s\n", (*nvmlErrorStringPtr)(nvret));
            return PAPI_EINVAL;
        }
    }
    break;

    default:
        return PAPI_EINVAL;
    }

    return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
_papi_nvml_init_thread(hwd_context_t * ctx)
{
    (void) ctx;

    SUBDBG("Enter: ctx: %p\n", ctx);

    return PAPI_OK;
}

static int
detectDevices()
{
    nvmlReturn_t ret;
    nvmlEnableState_t mode        = NVML_FEATURE_DISABLED;
    nvmlEnableState_t pendingmode = NVML_FEATURE_DISABLED;

    char name[64];
    char inforomECC[16];
    char names[device_count][64];

    float ecc_version = 0.0;

    int i = 0;

    unsigned int temp = 0;

    memset(names, 0x0, device_count * 64);

    /* So for each card, check whats querable */
    for (i = 0; i < device_count; i++) {
        features[i] = 0;
        
        ret = (*nvmlDeviceGetHandleByIndexPtr)(i, &devices[i]);
        if (NVML_SUCCESS != ret) {
            SUBDBG("nvmlDeviceGetHandleByIndex(%d, &devices[%d]) failed.\n", i, i);
            return PAPI_ESYS;
        }

        ret = (*nvmlDeviceGetNamePtr)(devices[i], name, sizeof(name) - 1);
        if (NVML_SUCCESS != ret) {
            SUBDBG("nvmlDeviceGetName failed \n");
            const char *name_unknown = "deviceNameUnknown";
            strncpy(name, name_unknown, strlen(name_unknown) + 1);
        }

        ret = (*nvmlDeviceGetInforomVersionPtr)(devices[i], NVML_INFOROM_ECC, inforomECC, 16);
        if (NVML_SUCCESS != ret) {
            SUBDBG("nvmlGetInforomVersion fails %s\n", (*nvmlErrorStringPtr)(ret));
        } else {
            ecc_version = strtof(inforomECC, NULL);
        }

        if (getClockSpeed(devices[i], NVML_CLOCK_GRAPHICS) != (unsigned long long) - 1) {
            features[i] |= FEATURE_CLOCK_INFO;
            num_events += 3;
        }

        /*  For Tesla and Quadro products from Fermi and Kepler families.
            requires NVML_INFOROM_ECC 2.0 or higher for location-based counts
            requires NVML_INFOROM_ECC 1.0 or higher for all other ECC counts
            requires ECC mode to be enabled. */
        ret = (*nvmlDeviceGetEccModePtr)(devices[i], &mode, &pendingmode);
        if (NVML_SUCCESS == ret) {
            if (NVML_FEATURE_ENABLED == mode) {
                if (ecc_version >= 2.0) {
                    features[i] |= FEATURE_ECC_LOCAL_ERRORS;
                    num_events += 8; /* {single bit, two bit errors} x { reg, l1, l2, memory } */
                }
                if (ecc_version >= 1.0) {
                    features[i] |= FEATURE_ECC_TOTAL_ERRORS;
                    num_events += 2; /* single bit errors, double bit errors */
                }
            }
        } else {
            SUBDBG("nvmlDeviceGetEccMode does not appear to be supported. (nvml return code %d)\n", ret);
        }

        /* Check if fan speed is available */
        if (getFanSpeed(devices[i]) != (unsigned long long) - 1) {
            features[i] |= FEATURE_FAN_SPEED;
            num_events++;
        }

        /* Check if clock data are available */
        if (getMaxClockSpeed(devices[i], NVML_CLOCK_GRAPHICS) != (unsigned long long) - 1) {
            features[i] |= FEATURE_MAX_CLOCK;
            num_events += 3;
        }

        /* For all products */
        features[i] |= FEATURE_MEMORY_INFO;
        num_events += 3; /* total, free, used */

        /* Check if performance state is available */
        if (getPState(devices[i]) != (unsigned long long) - 1) {
            features[i] |= FEATURE_PERF_STATES;
            num_events++;
        }

        /*  For "GF11x" Tesla and Quadro products from the Fermi family
                requires NVML_INFOROM_POWER 3.0 or higher
                For Tesla and Quadro products from the Kepler family
                does not require NVML_INFOROM_POWER */
        /* Just try reading power, if it works, enable it*/
        ret = (*nvmlDeviceGetPowerUsagePtr)(devices[i], &temp);
        if (NVML_SUCCESS == ret) {
            features[i] |= FEATURE_POWER;
            num_events++;
        } else {
            SUBDBG("nvmlDeviceGetPowerUsage does not appear to be supported on this card. (nvml return code %d)\n", ret);
        }

        /* Check if temperature data are available */
        if (getTemperature(devices[i]) != (unsigned long long) - 1) {
            features[i] |= FEATURE_TEMP;
            num_events++;
        }

        // For power_management_limit
        {
            // Just try the call to see if it works
            unsigned int templimit = 0;
            ret = (*nvmlDeviceGetPowerManagementLimitPtr)(devices[i], &templimit);
            if (ret == NVML_SUCCESS && templimit > 0) {
                power_management_initial_limit[i] = templimit;
                features[i] |= FEATURE_POWER_MANAGEMENT;
                num_events += 1;
            } else {
                power_management_initial_limit[i] = 0;
                SUBDBG("nvmlDeviceGetPowerManagementLimit not appear to be supported on this card. (NVML code %d)\n", ret);
            }
        }

        // For power_management_limit_constraints, minimum and maximum
        {
            unsigned int minLimit = 0, maxLimit = 0;
            ret = (*nvmlDeviceGetPowerManagementLimitConstraintsPtr)(devices[i], &minLimit, &maxLimit);
            if (ret == NVML_SUCCESS) {
                power_management_limit_constraint_min[i] = minLimit;
                features[i] |= FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN;
                num_events += 1;
                power_management_limit_constraint_max[i] = maxLimit;
                features[i] |= FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX;
                num_events += 1;
            } else {
                power_management_limit_constraint_min[i] = 0;
                power_management_limit_constraint_max[i] = INT_MAX;
            }
            SUBDBG("Done nvmlDeviceGetPowerManagementLimitConstraintsPtr\n");
        }

        /* Check if temperature data are available */
        if (getUtilization(devices[i], GPU_UTILIZATION) != (unsigned long long) - 1) {
            features[i] |= FEATURE_UTILIZATION;
            num_events += 2;
        }

        int retval = snprintf(names[i], sizeof(name), "%s:device:%d", name, i);
        if (retval > (int)sizeof(name)) {
            SUBDBG("Device name is too long %s:device%d", name, i);
            return (PAPI_EINVAL);
        }
        names[i][sizeof(name) - 1] = '\0';
    }
    return PAPI_OK;
}

static void
createNativeEvents()
{
    char name[PAPI_MIN_STR_LEN];
    char sanitized_name[PAPI_MIN_STR_LEN];
    char names[device_count][PAPI_MAX_STR_LEN];

    int i, nameLen = 0, j, devTableIdx = 0;

    nvml_native_event_entry_t* entry;
    nvmlReturn_t ret;

    nvml_native_table = (nvml_native_event_entry_t*) papi_malloc(
                            sizeof(nvml_native_event_entry_t) * num_events);
    memset(nvml_native_table, 0x0, sizeof(nvml_native_event_entry_t) * num_events);
    entry = &nvml_native_table[0];
    nvml_dev_id_table = (int*) papi_malloc(num_events*sizeof(int));

    for (i = 0; i < device_count; i++) {
        memset(names[i], 0x0, PAPI_MAX_STR_LEN);
        ret = (*nvmlDeviceGetNamePtr)(devices[i], name, sizeof(name) - 1);
        if (NVML_SUCCESS != ret) {
            SUBDBG("nvmlDeviceGetName failed \n");
            const char *name_unknown = "deviceNameUnknown";
            strncpy(name, name_unknown, strlen(name_unknown) + 1);
        }

        nameLen = strlen(name);
        strncpy(sanitized_name, name, PAPI_MIN_STR_LEN);

        int retval = snprintf(sanitized_name, sizeof(name), "%s:device_%d", name, i);
        if (retval > (int)sizeof(name)) {
            SUBDBG("Device name is too long %s:device%d", name, i);
            return;
        }
        sanitized_name[sizeof(name) - 1] = '\0';

        for (j = 0; j < nameLen; j++)
            if (' ' == sanitized_name[j])
                sanitized_name[j] = '_';

        if (HAS_FEATURE(features[i], FEATURE_CLOCK_INFO)) {
            sprintf(entry->name, "%s:graphics_clock", sanitized_name);
            strncpy(entry->description, "Graphics clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_GRAPHICS;
            entry->type = FEATURE_CLOCK_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:sm_clock", sanitized_name);
            strncpy(entry->description, "SM clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_SM;
            entry->type = FEATURE_CLOCK_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:memory_clock", sanitized_name);
            strncpy(entry->description, "Memory clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_MEM;
            entry->type = FEATURE_CLOCK_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_ECC_LOCAL_ERRORS)) {
            sprintf(entry->name, "%s:l1_single_ecc_errors", sanitized_name);
            strncpy(entry->description, "L1 cache single bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_SINGLE_BIT_ECC,
                 .which_one = LOCAL_ECC_L1,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:l2_single_ecc_errors", sanitized_name);
            strncpy(entry->description, "L2 cache single bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_SINGLE_BIT_ECC,
                 .which_one = LOCAL_ECC_L2,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:memory_single_ecc_errors", sanitized_name);
            strncpy(entry->description, "Device memory single bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_SINGLE_BIT_ECC,
                 .which_one = LOCAL_ECC_MEM,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:regfile_single_ecc_errors", sanitized_name);
            strncpy(entry->description, "Register file single bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_SINGLE_BIT_ECC,
                 .which_one = LOCAL_ECC_REGFILE,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:1l_double_ecc_errors", sanitized_name);
            strncpy(entry->description, "L1 cache double bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_DOUBLE_BIT_ECC,
                 .which_one = LOCAL_ECC_L1,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:l2_double_ecc_errors", sanitized_name);
            strncpy(entry->description, "L2 cache double bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_DOUBLE_BIT_ECC,
                 .which_one = LOCAL_ECC_L2,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:memory_double_ecc_errors", sanitized_name);
            strncpy(entry->description, "Device memory double bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_DOUBLE_BIT_ECC,
                 .which_one = LOCAL_ECC_MEM,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:regfile_double_ecc_errors", sanitized_name);
            strncpy(entry->description, "Register file double bit ECC", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_DOUBLE_BIT_ECC,
                 .which_one = LOCAL_ECC_REGFILE,
            };
            entry->type = FEATURE_ECC_LOCAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_FAN_SPEED)) {
            sprintf(entry->name, "%s:fan_speed", sanitized_name);
            strncpy(entry->description, "The fan speed expressed as a percent of the maximum, i.e. full speed is 100%", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_FAN_SPEED;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_MAX_CLOCK)) {
            sprintf(entry->name, "%s:graphics_max_clock", sanitized_name);
            strncpy(entry->description, "Maximal Graphics clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_GRAPHICS;
            entry->type = FEATURE_MAX_CLOCK;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:sm_max_clock", sanitized_name);
            strncpy(entry->description, "Maximal SM clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_SM;
            entry->type = FEATURE_MAX_CLOCK;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:memory_max_clock", sanitized_name);
            strncpy(entry->description, "Maximal Memory clock domain (MHz).", PAPI_MAX_STR_LEN);
            entry->options.clock = NVML_CLOCK_MEM;
            entry->type = FEATURE_MAX_CLOCK;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_MEMORY_INFO)) {
            sprintf(entry->name, "%s:total_memory", sanitized_name);
            strncpy(entry->description, "Total installed FB memory (in bytes).", PAPI_MAX_STR_LEN);
            entry->options.which_one = MEMINFO_TOTAL_MEMORY;
            entry->type = FEATURE_MEMORY_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:unallocated_memory", sanitized_name);
            strncpy(entry->description, "Uncallocated FB memory (in bytes).", PAPI_MAX_STR_LEN);
            entry->options.which_one = MEMINFO_UNALLOCED;
            entry->type = FEATURE_MEMORY_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:allocated_memory", sanitized_name);
            strncpy(entry->description, "Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping.", PAPI_MAX_STR_LEN);
            entry->options.which_one = MEMINFO_ALLOCED;
            entry->type = FEATURE_MEMORY_INFO;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_PERF_STATES)) {
            sprintf(entry->name, "%s:pstate", sanitized_name);
            strncpy(entry->description, "The performance state of the device.", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_PERF_STATES;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_POWER)) {
            sprintf(entry->name, "%s:power", sanitized_name);
            // set the power event units value to "mW" for miliwatts
            strncpy(entry->units, "mW", PAPI_MIN_STR_LEN);
            strncpy(entry->description, "Power usage reading for the device, in miliwatts. This is the power draw (+/-5 watts) for the entire board: GPU, memory, etc.", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_POWER;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_TEMP)) {
            sprintf(entry->name, "%s:temperature", sanitized_name);
            strncpy(entry->description, "Current temperature readings for the device, in degrees C.", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_TEMP;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_ECC_TOTAL_ERRORS)) {
            sprintf(entry->name, "%s:total_ecc_errors", sanitized_name);
            strncpy(entry->description, "Total single bit errors.", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_SINGLE_BIT_ECC,
            };
            entry->type = FEATURE_ECC_TOTAL_ERRORS;
            entry++;

            sprintf(entry->name, "%s:total_ecc_errors", sanitized_name);
            strncpy(entry->description, "Total double bit errors.", PAPI_MAX_STR_LEN);
            entry->options.ecc_opts = (struct local_ecc) {
                .bits = NVML_DOUBLE_BIT_ECC,
            };
            entry->type = FEATURE_ECC_TOTAL_ERRORS;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_UTILIZATION)) {
            sprintf(entry->name, "%s:gpu_utilization", sanitized_name);
            strncpy(entry->description, "Percent of time over the past second during which one or more kernels was executing on the GPU.", PAPI_MAX_STR_LEN);
            entry->options.which_one = GPU_UTILIZATION;
            entry->type = FEATURE_UTILIZATION;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;

            sprintf(entry->name, "%s:memory_utilization", sanitized_name);
            strncpy(entry->description, "Percent of time over the past second during which global (device) memory was being read or written.", PAPI_MAX_STR_LEN);
            entry->options.which_one = MEMORY_UTILIZATION;
            entry->type = FEATURE_UTILIZATION;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_POWER_MANAGEMENT)) {
            sprintf(entry->name, "%s:power_management_limit", sanitized_name);
            // set the power event units value to "mW" for milliwatts
            strncpy(entry->units, "mW", PAPI_MIN_STR_LEN);
            strncpy(entry->description, "Power draw upper bound limit (in mW) for the device. Writable (with appropriate privileges) on supported Kepler or later.", PAPI_MAX_STR_LEN - 1);
            entry->description[PAPI_MAX_STR_LEN - 1] = '\0';
            entry->type = FEATURE_POWER_MANAGEMENT;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }
        if (HAS_FEATURE(features[i], FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN)) {
            sprintf(entry->name, "%s:power_management_limit_constraint_min", sanitized_name);
            strncpy(entry->units, "mW", PAPI_MIN_STR_LEN);
            strncpy(entry->description, "The minimum power management limit in milliwatts.", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        if (HAS_FEATURE(features[i], FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX)) {
            sprintf(entry->name, "%s:power_management_limit_constraint_max", sanitized_name);
            strncpy(entry->units, "mW", PAPI_MIN_STR_LEN);
            strncpy(entry->description, "The maximum power management limit in milliwatts.", PAPI_MAX_STR_LEN);
            entry->type = FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX;
            entry++;
            nvml_dev_id_table[devTableIdx] = i;
            devTableIdx++;
        }

        strncpy(names[i], name, sizeof(names[0]) - 1);
        names[i][sizeof(names[0]) - 1] = '\0';
    }
} // create native events.


// Triggered by PAPI_shutdown(), but also if init fails to complete; for example due
// to a missing library. We still need to clean up. The dynamic libs (dlxxx routines)
// may have open mallocs that need to be free()d.
 
int _papi_nvml_shutdown_component()
{
    SUBDBG("Enter:\n");
    nvml_hardware_reset();
    if (nvml_native_table != NULL) papi_free(nvml_native_table);
    if (nvml_dev_id_table != NULL) papi_free(nvml_dev_id_table);
    if (devices != NULL) papi_free(devices);
    if (features != NULL) papi_free(features);
    if (power_management_initial_limit) papi_free(power_management_initial_limit);
    if (power_management_limit_constraint_min) papi_free(power_management_limit_constraint_min);
    if (power_management_limit_constraint_max) papi_free(power_management_limit_constraint_max);
    if (nvmlShutdownPtr) (*nvmlShutdownPtr)();        // Call nvml shutdown if we got that far.

    device_count = 0;
    num_events = 0;

    // close the dynamic libraries needed by this component (opened in the init component call)
    if (dl3) {dlclose(dl3); dl3=NULL;}

    return PAPI_OK;
}



/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */

static int _papi_nvml_init_component(int cidx)
{
    SUBDBG("Entry: cidx: %d\n", cidx);
    /* Export the total number of events available */
    _nvml_vector.cmp_info.num_native_events = -1;

    /* Export the component id */
    _nvml_vector.cmp_info.CmpIdx = cidx;

    /* Export the number of 'counters' */
    _nvml_vector.cmp_info.num_cntrs = -1;
    _nvml_vector.cmp_info.num_mpx_cntrs = -1;

     sprintf(_nvml_vector.cmp_info.disabled_reason,
             "Not initialized. Access component events to initialize it.");
    _nvml_vector.cmp_info.disabled = PAPI_EDELAY_INIT;

    return PAPI_EDELAY_INIT;
}

int _papi_nvml_init_private(void)
{
    nvmlReturn_t ret;
    int err = PAPI_OK;
    unsigned int nvml_count = 0;

    PAPI_lock(COMPONENT_LOCK);
    if (_nvml_vector.cmp_info.initialized) goto nvml_init_private_exit;

    SUBDBG("Private init with component idx: %d\n", _nvml_vector.cmp_info.CmpIdx);
    /* link in the NVML libraries and resolve the symbols we need to use */
    if (linkCudaLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of CUDA libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        _papi_nvml_shutdown_component();                          // clean up any open dynLibs, mallocs, etc.
        err = (PAPI_ENOSUPP);
        goto nvml_init_private_exit;
    }

    ret = (*nvmlInitPtr)();
    if (NVML_SUCCESS != ret) {
        strcpy(_nvml_vector.cmp_info.disabled_reason, "The NVIDIA management library failed to initialize.");
        _papi_nvml_shutdown_component();                          // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOSUPP;
        goto nvml_init_private_exit;
    }

    /* Figure out the number of CUDA devices in the system */
    ret = (*nvmlDeviceGetCountPtr)(&nvml_count);
    if (NVML_SUCCESS != ret) {
        strcpy(_nvml_vector.cmp_info.disabled_reason, "Unable to get a count of devices from the NVIDIA management library.");
        _papi_nvml_shutdown_component();                          // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOSUPP;
        goto nvml_init_private_exit;
    }

    device_count = nvml_count;
    SUBDBG("Need to setup NVML with %d devices\n", device_count);

    /* A per device representation of what events are present */
    features = (int*)papi_malloc(sizeof(int) * device_count);
    if (features == NULL) { 
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                    "%s failed to alloc %lu bytes for features.", __func__, sizeof(int)*device_count);
                    _nvml_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOMEM;
        goto nvml_init_private_exit;
    }

    /* Handles to each device */
    devices = (nvmlDevice_t*)papi_malloc(sizeof(nvmlDevice_t) * device_count);
    if (devices == NULL) {
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                    "%s failed to alloc %lu bytes for features.", __func__, (sizeof(nvmlDevice_t) * device_count));
                    _nvml_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOMEM;
        goto nvml_init_private_exit;
    }

    /* For each device, store the intial power value to enable reset if power is altered */
    power_management_initial_limit = (unsigned int*)papi_malloc(sizeof(unsigned int) * device_count);
    if (power_management_initial_limit == NULL) {
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                    "%s failed to alloc %lu bytes for power_management_initial_limit.", __func__, (sizeof(unsigned int) * device_count));
                    _nvml_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOMEM;
        goto nvml_init_private_exit;
    }
    power_management_limit_constraint_min = (unsigned int*)papi_malloc(sizeof(unsigned int) * device_count);
    if (power_management_limit_constraint_min == NULL) {
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                    "%s failed to alloc %lu bytes for power_management_limit_constraint_min.", __func__, (sizeof(unsigned int) * device_count));
                    _nvml_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOMEM;
        goto nvml_init_private_exit;
    }
    power_management_limit_constraint_max = (unsigned int*)papi_malloc(sizeof(unsigned int) * device_count);
    if (power_management_limit_constraint_max == NULL) {
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                    "%s failed to alloc %lu bytes for power_management_limit_constraint_max.", __func__, (sizeof(unsigned int) * device_count));
                    _nvml_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOMEM;
        goto nvml_init_private_exit;
    }

    /* Figure out what events are supported on each card. */
    if (detectDevices() != PAPI_OK) {
        sprintf(_nvml_vector.cmp_info.disabled_reason, "An error occured in device feature detection, please check your NVIDIA Management Library and CUDA install.");
        _papi_nvml_shutdown_component();                        // clean up any open dynLibs, mallocs, etc.
        err = PAPI_ENOSUPP;
        goto nvml_init_private_exit;
    }

    /* The assumption is that if everything went swimmingly in detectDevices,
        all nvml calls here should be fine. */
    createNativeEvents();

    /* Export the total number of events available */
    _nvml_vector.cmp_info.num_native_events = num_events;

    /* Export the number of 'counters' */
    _nvml_vector.cmp_info.num_cntrs = num_events;
    _nvml_vector.cmp_info.num_mpx_cntrs = num_events;

nvml_init_private_exit:
    _nvml_vector.cmp_info.initialized = 1;
    _nvml_vector.cmp_info.disabled = err;

    PAPI_unlock(COMPONENT_LOCK);

    return err;
}

/*
 * Link the necessary CUDA libraries to use the NVML component.  If any of them can not be found, then
 * the NVML component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the NVML component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */
static int
linkCudaLibraries()
{
    char path_lib[1024];
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if (_dl_non_dynamic_init != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML component does not support statically linking of libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }

    // Need to link in the NVML libraries, if any not found disable the component.
    // getenv returns NULL if environment variable is not found.
    char *cuda_root = getenv("PAPI_CUDA_ROOT");

    // We need the NVML main library, normally libnvidia-ml.so. 
    dl3 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.   
    if (strlen(nvml_main) > 0) {                                        // If override given, it MUST work.
        dl3 = dlopen(nvml_main, RTLD_NOW | RTLD_GLOBAL);                // Try to open that path.
        if (dl3 == NULL) {
            snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_NVML_MAIN override '%s' given in Rules.nvml not found.", nvml_main);
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl3 == NULL) {                                              // If no override,
        dl3 = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);    // Try system paths.
    }

    // Step 3: Try the explicit install default. 
    if (dl3 == NULL && cuda_root != NULL) {                                         // If ROOT given, it doesn't HAVE to work.
        snprintf(path_lib, 1024, "%s/lib64/libnvidia-ml.so", cuda_root);            // PAPI Root check.
        dl3 = dlopen(path_lib, RTLD_NOW | RTLD_GLOBAL);                             // Try to open that path.
    }

    // Check for failure.
    if (dl3 == NULL) {
        snprintf(_nvml_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libnvidia-ml.so not found.");
        return(PAPI_ENOSUPP);   // Not found on default paths.
    }

    // We have a dl3. (libnvidia-ml.so).

    nvmlDeviceGetClockInfoPtr = dlsym(dl3, "nvmlDeviceGetClockInfo");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetClockInfo not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlErrorStringPtr = dlsym(dl3, "nvmlErrorString");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlErrorString not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetDetailedEccErrorsPtr = dlsym(dl3, "nvmlDeviceGetDetailedEccErrors");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetDetailedEccErrors not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetFanSpeedPtr = dlsym(dl3, "nvmlDeviceGetFanSpeed");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetFanSpeed not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetMemoryInfoPtr = dlsym(dl3, "nvmlDeviceGetMemoryInfo");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetMemoryInfo not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetPerformanceStatePtr = dlsym(dl3, "nvmlDeviceGetPerformanceState");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetPerformanceState not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetPowerUsagePtr = dlsym(dl3, "nvmlDeviceGetPowerUsage");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetPowerUsage not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetTemperaturePtr = dlsym(dl3, "nvmlDeviceGetTemperature");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetTemperature not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetTotalEccErrorsPtr = dlsym(dl3, "nvmlDeviceGetTotalEccErrors");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetTotalEccErrors not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetUtilizationRatesPtr = dlsym(dl3, "nvmlDeviceGetUtilizationRates");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetUtilizationRates not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetHandleByIndexPtr = dlsym(dl3, "nvmlDeviceGetHandleByIndex");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetHandleByIndex not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetPciInfoPtr = dlsym(dl3, "nvmlDeviceGetPciInfo");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetPciInfo not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetNamePtr = dlsym(dl3, "nvmlDeviceGetName");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetName not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetInforomVersionPtr = dlsym(dl3, "nvmlDeviceGetInforomVersion");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetInforomVersion not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetEccModePtr = dlsym(dl3, "nvmlDeviceGetEccMode");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetEccMode not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlInitPtr = dlsym(dl3, "nvmlInit");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlInit not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetCountPtr = dlsym(dl3, "nvmlDeviceGetCount");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetCount not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlShutdownPtr = dlsym(dl3, "nvmlShutdown");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlShutdown not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetPowerManagementLimitPtr = dlsym(dl3, "nvmlDeviceGetPowerManagementLimit");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetPowerManagementLimit not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceSetPowerManagementLimitPtr = dlsym(dl3, "nvmlDeviceSetPowerManagementLimit");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceSetPowerManagementLimit not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    nvmlDeviceGetPowerManagementLimitConstraintsPtr = dlsym(dl3, "nvmlDeviceGetPowerManagementLimitConstraints");
    if (dlerror() != NULL) {
        strncpy(_nvml_vector.cmp_info.disabled_reason, "NVML function nvmlDeviceGetPowerManagementLimitConstraints not found.", PAPI_MAX_STR_LEN);
        return (PAPI_ENOSUPP);
    }
    return (PAPI_OK);
}

/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

int
_papi_nvml_init_control_state(hwd_control_state_t * ctl)
{
    SUBDBG("nvml_init_control_state... %p\n", ctl);
    DO_SOME_CHECKING(&_nvml_vector);
    nvml_control_state_t *nvml_ctl = (nvml_control_state_t *) ctl;
    memset(nvml_ctl, 0, sizeof(nvml_control_state_t));

    return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int
_papi_nvml_update_control_state(hwd_control_state_t *ctl,
                                NativeInfo_t *native,
                                int count,
                                hwd_context_t *ctx)
{
    SUBDBG("Enter: ctl: %p, ctx: %p\n", ctl, ctx);
    int i, index;

    nvml_control_state_t *nvml_ctl = (nvml_control_state_t *) ctl;
    (void) ctx;

    DO_SOME_CHECKING(&_nvml_vector);
    /* if no events, return */
    if (count == 0) return PAPI_OK;

    for (i = 0; i < count; i++) {
        index = native[i].ni_event;
        nvml_ctl->which_counter[i] = index;
        /* We have no constraints on event position, so any event */
        /* can be in any slot.                                    */
        native[i].ni_position = i;
    }
    nvml_ctl->num_events = count;
    return PAPI_OK;
}
/** Triggered by PAPI_start() */
int
_papi_nvml_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    SUBDBG("Enter: ctx: %p, ctl: %p\n", ctx, ctl);

    (void) ctx;
    (void) ctl;

    /* anything that would need to be set at counter start time */

    /* reset */
    /* start the counting */

    return PAPI_OK;
}

/** Triggered by PAPI_stop() */
int
_papi_nvml_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    SUBDBG("Enter: ctx: %p, ctl: %p\n", ctx, ctl);

    int i;
    (void) ctx;
    (void) ctl;
    int ret;

    nvml_control_state_t* nvml_ctl = (nvml_control_state_t*) ctl;

    for (i = 0; i < nvml_ctl->num_events; i++) {
        if (PAPI_OK !=
                (ret = nvml_hardware_read(&nvml_ctl->counter[i],
                                          nvml_ctl->which_counter[i])))
            return ret;

    }

    return PAPI_OK;
}

/** Triggered by PAPI_read() */
int
_papi_nvml_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
                long long **events, int flags)
{
    SUBDBG("Enter: ctx: %p, flags: %d\n", ctx, flags);

    (void) ctx;
    (void) flags;
    int i;
    int ret;
    nvml_control_state_t* nvml_ctl = (nvml_control_state_t*) ctl;

    for (i = 0; i < nvml_ctl->num_events; i++) {
        if (PAPI_OK !=
                (ret = nvml_hardware_read(&nvml_ctl->counter[i],
                                          nvml_ctl->which_counter[i])))
            return ret;

    }
    /* return pointer to the values we read */
    *events = nvml_ctl->counter;
    return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
_papi_nvml_write(hwd_context_t *ctx, hwd_control_state_t *ctl, long long *events)
{
    SUBDBG("Enter: ctx: %p, ctl: %p\n", ctx, ctl);
    (void) ctx;
    nvml_control_state_t* nvml_ctl = (nvml_control_state_t*) ctl;
    int i;
    int ret;

    /* You can change ECC mode and compute exclusivity modes on the cards */
    /* But I don't see this as a function of a PAPI component at this time */
    /* All implementation issues aside. */

    // Currently POWER_MANAGEMENT can be written
    for (i = 0; i < nvml_ctl->num_events; i++) {
        if (PAPI_OK != (ret = nvml_hardware_write(&events[i], nvml_ctl->which_counter[i])))
            return ret;
    }

    /* return pointer to the values we read */
    return PAPI_OK;
}

/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
int
_papi_nvml_reset(hwd_context_t * ctx, hwd_control_state_t * ctl)
{
    SUBDBG("Enter: ctx: %p, ctl: %p\n", ctx, ctl);

    (void) ctx;
    (void) ctl;

    /* Reset the hardware */
    nvml_hardware_reset();

    return PAPI_OK;
}

/** Called at thread shutdown */
int
_papi_nvml_shutdown_thread(hwd_context_t *ctx)
{
    SUBDBG("Enter: ctx: %p\n", ctx);

    (void) ctx;

    /* Last chance to clean up thread */

    return PAPI_OK;
}

/** This function sets various options in the component
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
int
_papi_nvml_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    SUBDBG("Enter: ctx: %p, code: %d\n", ctx, code);

    (void) ctx;
    (void) code;
    (void) option;

    /* FIXME.  This should maybe set up more state, such as which counters are active and */
    /*         counter mappings. */

    return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
  In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
  By default return PAPI_EINVAL if none of those are specified
  and PAPI_OK with success
  PAPI_DOM_USER is only user context is counted
  PAPI_DOM_KERNEL is only the Kernel/OS context is counted
  PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
  PAPI_DOM_ALL   is all of the domains
 */
int
_papi_nvml_set_domain(hwd_control_state_t * cntrl, int domain)
{
    SUBDBG("Enter: cntrl: %p, domain: %d\n", cntrl, domain);

    (void) cntrl;

    int found = 0;

    if (PAPI_DOM_USER & domain) {
        SUBDBG(" PAPI_DOM_USER \n");
        found = 1;
    }
    if (PAPI_DOM_KERNEL & domain) {
        SUBDBG(" PAPI_DOM_KERNEL \n");
        found = 1;
    }
    if (PAPI_DOM_OTHER & domain) {
        SUBDBG(" PAPI_DOM_OTHER \n");
        found = 1;
    }
    if (PAPI_DOM_ALL & domain) {
        SUBDBG(" PAPI_DOM_ALL \n");
        found = 1;
    }
    if (!found)
        return (PAPI_EINVAL);

    return PAPI_OK;
}

/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/

/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
int
_papi_nvml_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    int index;

    DO_SOME_CHECKING(&_nvml_vector);

    switch (modifier) {

    /* return EventCode of first event */
    case PAPI_ENUM_FIRST:
        /* return the first event that we support */

        *EventCode = 0;
        return PAPI_OK;

    /* return EventCode of next available event */
    case PAPI_ENUM_EVENTS:
        index = *EventCode;

        /* Make sure we are in range */
        if (index < num_events - 1) {

            /* This assumes a non-sparse mapping of the events */
            *EventCode = *EventCode + 1;
            return PAPI_OK;
        } else {
            return PAPI_ENOEVNT;
        }
        break;

    default:
        return PAPI_EINVAL;
    }

    return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
int
_papi_nvml_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    SUBDBG("Entry: EventCode: %#x, name: %s, len: %d\n", EventCode, name, len);
    int index;

    DO_SOME_CHECKING(&_nvml_vector);

    index = EventCode;

    /* Make sure we are in range */
    if (index >= num_events) return PAPI_ENOEVNT;

    strncpy(name, nvml_native_table[index].name, len);

    return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
int
_papi_nvml_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
    int index;
    index = EventCode;

    if (index >= num_events) return PAPI_ENOEVNT;

    strncpy(descr, nvml_native_table[index].description, len);

    return PAPI_OK;
}

/** Takes a native event code and passes back the event info
 * @param EventCode is the native event code
 * @param info is a pointer for the info to be copied to
 */
int
_papi_nvml_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{

    int index = EventCode;

    if ((index < 0) || (index >= num_events)) return PAPI_ENOEVNT;

    strncpy(info->symbol, nvml_native_table[index].name, sizeof(info->symbol) - 1);
    info->symbol[sizeof(info->symbol) - 1] = '\0';

    strncpy(info->units, nvml_native_table[index].units, sizeof(info->units) - 1);
    info->units[sizeof(info->units) - 1] = '\0';

    strncpy(info->long_descr, nvml_native_table[index].description, sizeof(info->long_descr) - 1);
    info->long_descr[sizeof(info->long_descr) - 1] = '\0';

//  info->data_type = nvml_native_table[index].return_type;

    return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _nvml_vector = {
    .cmp_info = {
        /* default component information */
        /* (unspecified values are initialized to 0) */

        .name = "nvml",
        .short_name = "nvml",
        .version = "1.0",
        .description = "NVML provides the API for monitoring NVIDIA hardware (power usage, temperature, fan speed, etc)",
        .support_version = "n/a",
        .kernel_version = "n/a",

        .num_preset_events = 0,
        .num_native_events = 0, /* set by init_component */
        .default_domain = PAPI_DOM_USER,
        .available_domains = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
        .hardware_intr = 0,
        .precise_intr = 0,
        .posix1b_timers = 0,
        .kernel_profile = 0,
        .kernel_multiplex = 0,
        .fast_counter_read = 0,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .cntr_umasks = 0,
        .cpu = 0,
        .inherit = 0,
        .initialized = 0,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof(nvml_context_t),
        .control_state = sizeof(nvml_control_state_t),
        .reg_value = sizeof(nvml_register_t),
        // .reg_alloc = sizeof ( nvml_reg_alloc_t ),
    },

    /* function pointers */

    /* Used for general PAPI interactions */
    .start =                _papi_nvml_start,
    .stop =                 _papi_nvml_stop,
    .read =                 _papi_nvml_read,
    .reset =                _papi_nvml_reset,
    .write =                _papi_nvml_write,
    .init_component =       _papi_nvml_init_component,
    .init_thread =          _papi_nvml_init_thread,
    .init_control_state =   _papi_nvml_init_control_state,
    .update_control_state = _papi_nvml_update_control_state,
    .ctl =                  _papi_nvml_ctl,
    .shutdown_thread =      _papi_nvml_shutdown_thread,
    .shutdown_component =   _papi_nvml_shutdown_component,
    .set_domain =           _papi_nvml_set_domain,
    .cleanup_eventset =     NULL,
    /* called in add_native_events() */
    .allocate_registers =   NULL,

    /* Used for overflow/profiling */
    .dispatch_timer =       NULL,
    .get_overflow_address = NULL,
    .stop_profiling =       NULL,
    .set_overflow =         NULL,
    .set_profile =          NULL,

    /* Name Mapping Functions */
    .ntv_enum_events =   _papi_nvml_ntv_enum_events,
    .ntv_name_to_code  = NULL,
    .ntv_code_to_name =  _papi_nvml_ntv_code_to_name,
    .ntv_code_to_descr = _papi_nvml_ntv_code_to_descr,
    .ntv_code_to_info = _papi_nvml_ntv_code_to_info,

};

