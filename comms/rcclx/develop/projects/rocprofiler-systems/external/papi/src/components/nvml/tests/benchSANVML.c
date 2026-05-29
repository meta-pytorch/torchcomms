//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// This bench interacts directly with NVML from a main(). The objective here is
// to test reading and setting of a power limit without PAPI, to determine if 
// errors are PAPI related or not. It can be modified to test other NVML
// events.
//
// Much of this code is scavenged from linux-nvml.c. 
// Author: Tony Castaldo (tonycastaldo@icl.utk.edu).
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <dlfcn.h>      // Dynamic lib routines; especially dlsym to get func ptrs.

#include "nvml.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

/*****  CHANGE PROTOTYPES TO DECLARE CUDA AND NVML LIBRARY SYMBOLS AS WEAK  *****
 *  This is done so that a version of PAPI built with the nvml component can    *
 *  be installed on a system which does not have the cuda libraries installed.  *
 *                                                                              *
 *  If this is done without these prototypes, then all papi services on the     *
 *  system without the cuda libraries installed will fail.  The PAPI libraries  *
 *  contain references to the cuda libraries which are not installed.  The      *
 *  load of PAPI commands fails because the cuda library references can not be  *
 *  resolved.                                                                   *
 *                                                                              *
 *  This also defines pointers to the cuda library functions that we call.      *
 *  These function pointers will be resolved with dlopen/dlsym calls at         *
 *  component initialization time.  The component then calls the cuda library   *
 *  functions through these function pointers.                                  *
 ********************************************************************************/
#undef CUDAAPI
#define CUDAAPI __attribute__((weak))
CUresult CUDAAPI (*cuInitPtr)(unsigned int);
CUresult CUDAAPI cuInit(unsigned int myInt) {return (*cuInitPtr)(myInt);}

#undef CUDARTAPI
#define CUDARTAPI __attribute__((weak))

cudaError_t (*cudaGetDevicePtr)(int *);
cudaError_t (*cudaGetDeviceCountPtr)(int *);
cudaError_t (*cudaDeviceGetPCIBusIdPtr)(char *, int, int);

cudaError_t CUDARTAPI cudaGetDevice(int *dest) {return (*cudaGetDevicePtr)(dest); };
cudaError_t CUDARTAPI cudaGetDeviceCount(int *dest) {return (*cudaGetDeviceCountPtr)(dest);}
cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *, int, int);

#undef DECLDIR
#define DECLDIR __attribute__((weak))

char* (*nvmlErrorStringPtr)(nvmlReturn_t);
nvmlReturn_t (*nvmlDeviceGetClockInfoPtr)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
nvmlReturn_t (*nvmlDeviceGetCountPtr)(unsigned int *dest);
nvmlReturn_t (*nvmlDeviceGetDetailedEccErrorsPtr)(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *);
nvmlReturn_t (*nvmlDeviceGetEccModePtr)(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
nvmlReturn_t (*nvmlDeviceGetFanSpeedPtr)(nvmlDevice_t, unsigned int *);
nvmlReturn_t (*nvmlDeviceGetHandleByIndexPtr)(unsigned int, nvmlDevice_t *);
nvmlReturn_t (*nvmlDeviceGetInforomVersionPtr)(nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int);
nvmlReturn_t (*nvmlDeviceGetMemoryInfoPtr)(nvmlDevice_t, nvmlMemory_t *);
nvmlReturn_t (*nvmlDeviceGetNamePtr)(nvmlDevice_t, char *, unsigned int);
nvmlReturn_t (*nvmlDeviceGetPciInfoPtr)(nvmlDevice_t, nvmlPciInfo_t *);
nvmlReturn_t (*nvmlDeviceGetPerformanceStatePtr)(nvmlDevice_t, nvmlPstates_t *);
nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitConstraintsPtr)(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit);
nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitPtr)(nvmlDevice_t device, unsigned int* limit);
nvmlReturn_t (*nvmlDeviceGetPowerUsagePtr)(nvmlDevice_t, unsigned int *);
nvmlReturn_t (*nvmlDeviceGetTemperaturePtr)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
nvmlReturn_t (*nvmlDeviceGetTotalEccErrorsPtr)(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, unsigned long long *);
nvmlReturn_t (*nvmlDeviceGetUtilizationRatesPtr)(nvmlDevice_t, nvmlUtilization_t *);
nvmlReturn_t (*nvmlDeviceSetPowerManagementLimitPtr)(nvmlDevice_t device, unsigned int  limit);
nvmlReturn_t (*nvmlInitPtr)(void);
nvmlReturn_t (*nvmlShutdownPtr)(void);

const char*  DECLDIR nvmlErrorString(nvmlReturn_t);
nvmlReturn_t DECLDIR nvmlDeviceGetClockInfo(nvmlDevice_t, nvmlClockType_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetCount(unsigned int *dest){return (*nvmlDeviceGetCountPtr)(dest);}
nvmlReturn_t DECLDIR nvmlDeviceGetDetailedEccErrors(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetEccMode(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetFanSpeed(nvmlDevice_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByIndex(unsigned int idx, nvmlDevice_t *dest) {return (*nvmlDeviceGetHandleByIndexPtr)(idx, dest); }
nvmlReturn_t DECLDIR nvmlDeviceGetInforomVersion(nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int);
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo(nvmlDevice_t, nvmlMemory_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int len) {(*nvmlDeviceGetNamePtr)(device, name, len); }
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo(nvmlDevice_t, nvmlPciInfo_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetPerformanceState(nvmlDevice_t, nvmlPstates_t *);
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) {
                     (*nvmlDeviceGetPowerManagementLimitPtr)(device, limit); }
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) {
                     (*nvmlDeviceGetPowerManagementLimitConstraintsPtr)(device, minLimit, maxLimit); }
nvmlReturn_t DECLDIR nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *dest) {
                     (*nvmlDeviceGetPowerUsagePtr)(device, dest); }
nvmlReturn_t DECLDIR nvmlDeviceGetTemperature(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
nvmlReturn_t DECLDIR nvmlDeviceGetTotalEccErrors(nvmlDevice_t, nvmlEccBitType_t, nvmlEccCounterType_t, unsigned long long *);
nvmlReturn_t DECLDIR nvmlDeviceGetUtilizationRates(nvmlDevice_t, nvmlUtilization_t *);
nvmlReturn_t DECLDIR nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int  limit) {
                     (*nvmlDeviceSetPowerManagementLimitPtr)(device, limit); }
nvmlReturn_t DECLDIR nvmlInit(void){return (*nvmlInitPtr)();}
nvmlReturn_t DECLDIR nvmlShutdown(void);

// file handles used to access cuda libraries with dlopen
static void* dl1 = NULL;
static void* dl2 = NULL;
static void* dl3 = NULL;

static struct timeval t1, t2;                                        // used in timing routines to measure performance.

//-----------------------------------------------------------------------------
// Union to convert pointers and avoid warnings. Plug in one, pull out the other.
//-----------------------------------------------------------------------------
typedef union  
{
   void                 *vPtr;
   int                  *iPtr;
   unsigned int         *uiPtr;
   long                 *lPtr;
   long long            *llPtr;
   unsigned long long   *ullPtr;
   float                *fPtr;
   double               *dPtr;
   char                 *cPtr;
} uPointer_t;

typedef union
{
   long long ll;
   unsigned long long ull;
   double    d;
   void *vp;
   unsigned char ch[8];
} convert_64_t;


// -------------------------- GLOBAL SECTION ---------------------------------

//--------------------------------------------------------------------
// Timing of routines and blocks. Typical usage;
// gettimeofday(&t1, NULL);                  // starting point.
// ... some code to execute ...
// gettimeofday(&t2, NULL);                  // finished timing.
// fprintf(stderr, "routine took %li uS.\n", // report time.
//                       (mConvertUsec(t2)-mConvertUsec(t1)));
#define _prog_fprintf if (1) fprintf                                    /* change to 1 to enable printing of progress debug messages. TURN OFF if benchmark timing.    */
#define _time_fprintf if (1) fprintf                                    /* change to 1 to enable printing of performance timings.     TURN OFF if benchmark timing.    */

//-----------------------------------------------------------------------------
// Using weak symbols (global declared without a value, so it defers to any
// other global declared in another file WITH a value) allows PAPI to be built
// with the component, but PAPI can still be installed in a system without the
// required library.
//-----------------------------------------------------------------------------

void (*_dl_non_dynamic_init)(void) __attribute__((weak));               // declare a weak dynamic-library init routine pointer.

/** Number of devices detected at component_init time */
static int device_count = 0;

static nvmlDevice_t* devices = NULL;
static int* features = NULL;
static unsigned int *power_management_initial_limit = NULL;
static unsigned int *power_management_limit_constraint_min = NULL;
static unsigned int *power_management_limit_constraint_max = NULL;

//-----------------------------------------------------------------------------
// Get all needed function pointers from the Dynamic Link Library. 
//-----------------------------------------------------------------------------

// MACRO checks for Dynamic Lib failure, reports, returns Not Supported.
#define mCheck_DL_Status( err, str )                                          \
   if( err )                                                                  \
   {                                                                          \
      fprintf(stderr, str);                                                   \
      return(-1);                                                             \
   }

// keys for above: Init, InitThrd, InitCtlSt, Stop, ShutdownThrd, ShutdownCmp, Start,
// UpdateCtl, Read, Ctl, SetDom, Reset, Enum, EnumFirst, EnumNext, EnumUmasks, 
// NameToCode, CodeToName, CodeToDesc, CodeToInfo.

// Simplify routine below; relies on ptr names being same as func tags.
#define STRINGIFY(x) #x 
#define TOSTRING(x) STRINGIFY(x)
#define mGet_DL_FPtr(libPtr, Name)                                         \
   Name##Ptr = dlsym(libPtr, TOSTRING(Name));                              \
   mCheck_DL_Status(dlerror()!=NULL, TOSTRING(libPtr) " Library function " \
                  TOSTRING(Name) " not found.");

int _local_linkDynamicLibraries(void) 
{
   if (_dl_non_dynamic_init != NULL) {    // If weak var is present, we are statically linked instead of dynamically.
      fprintf(stderr, "NVML component does not support statically linked libc.");
      return (-1);
   }

   // Exit if we cannot link the cuda or NVML libs.
   dl1 = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
   if (!dl1) {
      fprintf(stderr, "CUDA library libcuda.so not found.");
      return (-1);
   }

   dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
   if (!dl2) {
      fprintf(stderr, "CUDA runtime library libcudart.so not found.");
      return (-1);
   }

   dl3 = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
   if (!dl3) {
      fprintf(stderr, "NVML runtime library libnvidia-ml.so not found.");
       return (-1);
   }

//-----------------------------------------------------------------------------
// Collect pointers for routines in shared library.  All below will abort this
// routine with -1, the routine is not found in the dynamic library.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Collect pointers for routines in shared library.  All below will abort this
// routine with -1, the routine is not found in the dynamic library.
//-----------------------------------------------------------------------------

   mGet_DL_FPtr(dl1, cuInit);

   mGet_DL_FPtr(dl2, cudaGetDevice);
   mGet_DL_FPtr(dl2, cudaGetDeviceCount);
   mGet_DL_FPtr(dl2, cudaDeviceGetPCIBusId);

   mGet_DL_FPtr(dl3, nvmlDeviceGetClockInfo);
   mGet_DL_FPtr(dl3, nvmlErrorString);
   mGet_DL_FPtr(dl3, nvmlDeviceGetDetailedEccErrors);
   mGet_DL_FPtr(dl3, nvmlDeviceGetFanSpeed);
   mGet_DL_FPtr(dl3, nvmlDeviceGetMemoryInfo);
   mGet_DL_FPtr(dl3, nvmlDeviceGetPerformanceState);
   mGet_DL_FPtr(dl3, nvmlDeviceGetPowerUsage);
   mGet_DL_FPtr(dl3, nvmlDeviceGetTemperature);
   mGet_DL_FPtr(dl3, nvmlDeviceGetTotalEccErrors);
   mGet_DL_FPtr(dl3, nvmlDeviceGetUtilizationRates);
   mGet_DL_FPtr(dl3, nvmlDeviceGetHandleByIndex);
   mGet_DL_FPtr(dl3, nvmlDeviceGetPciInfo);
   mGet_DL_FPtr(dl3, nvmlDeviceGetName);
   mGet_DL_FPtr(dl3, nvmlDeviceGetInforomVersion);
   mGet_DL_FPtr(dl3, nvmlDeviceGetEccMode);
   mGet_DL_FPtr(dl3, nvmlInit);
   mGet_DL_FPtr(dl3, nvmlDeviceGetCount);
   mGet_DL_FPtr(dl3, nvmlShutdown);
   mGet_DL_FPtr(dl3, nvmlDeviceGetPowerManagementLimit);
   mGet_DL_FPtr(dl3, nvmlDeviceSetPowerManagementLimit);
   mGet_DL_FPtr(dl3, nvmlDeviceGetPowerManagementLimitConstraints);

   return 0;         // If we get here, all above succeeded. 
} // end routine.

//----------------------------------------------------------------------------
// main(). intialize the lib, then work on reading the value. 
//---------------------------------------------------------------------------
int main (int argc, char **argv) 
{
   (void) argc; (void) argv;                                            // Prevent not used warning.
   #define hostnameLen 512 /* constant used multiple times. */
   char hostname[hostnameLen];                                          // host name.
   int  i, j, ret;
   int cuda_count, nvml_count;
    nvmlReturn_t nvret;
    cudaError_t cuerr;

   //-------------------------------------------------------------------
   // Begin initialization timing.
   //-------------------------------------------------------------------

   gettimeofday(&t1, NULL);
   ret = _local_linkDynamicLibraries();
   if ( ret != 0) {                                                     // Failure to get lib.
      fprintf(stderr, "Failed attempt to link to CUDA and NVML libraries.");
      exit(-1); 
   }

   _prog_fprintf(stderr, "Linked to CUDA and NVML libraries\n");        // debug only; turn off if timing.
   
   nvret = nvmlInit();                                                  // Initialize the library.
   if (nvret != NVML_SUCCESS) {
      fprintf(stderr, "Failed nvmlInit(), ret=%i [%s].\n", nvret, nvmlErrorString(nvret));
      exit(-1);
   }

   nvret = cuInit(0);                                                   // Initialize the CUDA library.
   if (nvret != cudaSuccess) {
      fprintf(stderr, "Failed cuInit(0).\n");
      exit(-1);
   }

   nvret = nvmlDeviceGetCount(&nvml_count);                             // Get the device count.
   if (nvret != NVML_SUCCESS) {
      fprintf(stderr, "nvmlDeviceGetCount failed; ret=%i.\n", nvret);   // Report an error.
      exit(-1);
   }

   nvret = cudaGetDeviceCount(&cuda_count);                             // Get the device count.
   if (nvret != cudaSuccess) {
      fprintf(stderr, "cudaGetDeviceCount failed; ret=%i.\n", nvret);   // Report an error.
      exit(-1);
   }


   ret = gethostname(hostname, hostnameLen);                            // Try to get the host hame.
   if( gethostname(hostname, hostnameLen) != 0) {                       // If we can't get the hostname, 
      fprintf(stderr, "Failed system call, gethostname() "
            "returned %i.", ret);
   exit(-1);
   }
   #undef hostnameLen /* done with it. */

   fprintf(stderr, "hostname: %s\n"
                   "nvml_count=%i\n"
                   "cuda_count=%i\n", hostname, nvml_count, cuda_count);

   nvmlDevice_t *handle = malloc(nvml_count * sizeof(nvmlDevice_t));    // for all device handles.
   char name[128];                                                      // space for device name.
   unsigned int powerUsage, powerLimit, powerLimit2;                    // for the power usage and limit.
   unsigned int minPower, maxPower;                                     // Minimum and Maximum power.

   // scan all the devices.
   for (i=0; i<nvml_count; i++) {                                       // Get all the handles; print as we go.
      nvret = nvmlDeviceGetHandleByIndex(i, &handle[i]);                // Read the handle.
      if (nvret != NVML_SUCCESS) {
         fprintf(stderr, "nvmlDeviceGetHandleByIndex %i failed; nvret=%i [%s].\n", i, nvret, nvmlErrorString(nvret));
         handle[i]=NULL;                                                // Set to bad value. 
         continue;                                                      // skip trying this one.
      }
 
      fprintf(stderr, "Handle %i: %016lX\n", i, handle[i]);             // Show the handles.

      nvret = nvmlDeviceGetName(handle[i], name, sizeof(name)-1);       // Get the name.
      name[sizeof(name)-1]='\0';                                        // Ensure z-termination.
      fprintf(stderr, "Name='%s'.\n", name);                            // Show the name.

      nvret = nvmlDeviceGetPowerUsage(handle[i], &powerUsage);          // Attempt to get power usage.
      if (nvret != NVML_SUCCESS) {                                      // If it failed,
         fprintf(stderr, "nvmlDeviceGetPowerUsage failed; nvret=%i [%s]\n", nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceGetPowerUsage succeeded, value returned=%u mw.\n", powerUsage);
      }

      nvret = nvmlDeviceGetPowerManagementLimit(handle[i], &powerLimit);// Attempt to get power limit.
      if (nvret != NVML_SUCCESS) {                                      // If it failed,
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimit failed; nvret=%i [%s]\n", nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimit succeeded, value returned=%u mw.\n", powerLimit);
      }

      nvret = nvmlDeviceGetPowerManagementLimitConstraints(handle[i], &minPower, &maxPower);// Attempt to get min and max of power limit.
      if (nvret != NVML_SUCCESS) {                                      // If it failed,
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimitConstraints failed; nvret=%i [%s]\n", nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimitConstraints succeeded, values min=%u mw, max=%u mw.\n", minPower, maxPower);
      }

      // Test setting the power, to top-100.
      unsigned int newPower=maxPower-100;                                     // compute a new power setting.
      nvret = nvmlDeviceSetPowerManagementLimit(handle[i], newPower);         // Attempt to set it.
      if (nvret != NVML_SUCCESS) {                                            // If it failed,
         fprintf(stderr, "nvmlDeviceSetPowerManagementLimit to %i failed; nvret=%i [%s]\n", newPower, nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceSetPowerManagementLimit to %i succeeded. (Routine call did not return error).\n", newPower);
      }

      nvret = nvmlDeviceGetPowerManagementLimit(handle[i], &powerLimit2);     // Attempt to get new power limit.
      if (nvret != NVML_SUCCESS) {                                            // If it failed,
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimit failed; nvret=%i [%s]\n", nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceGetPowerManagementLimit call to check setting succeeded, value returned=%u mw.\n", powerLimit2);
         if (powerLimit2 != newPower) {
            fprintf(stderr, "Note the check failed, the limit read is not the limit we tried to set.\n");
         } else {
            fprintf(stderr, "Note the check is a success, the power limit was changed.\n");
         }
      }

      nvret = nvmlDeviceSetPowerManagementLimit(handle[i], powerLimit);       // In case it works, set it back to where we found it.
      if (nvret != NVML_SUCCESS) {                                            // If it failed,
         fprintf(stderr, "nvmlDeviceSetPowerManagementLimit to restore %i failed; nvret=%i [%s]\n", powerLimit, nvret, nvmlErrorString(nvret));
      } else {
         fprintf(stderr, "nvmlDeviceSetPowerManagementLimit to restore %i succeeded.\n", powerLimit);
      }






   } // end of loop through devices.
 
   
   
   
   //-------------------------------------------------------------------
   // Cleanup, and shutdown.
   //-------------------------------------------------------------------

   return 0;
} // end MAIN routine.


