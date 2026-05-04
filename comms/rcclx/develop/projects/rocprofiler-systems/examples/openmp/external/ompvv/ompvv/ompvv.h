//===------ ompvv.h ------------------ OMPVV HEADER FILE ------------------===//
//
// Header file for OMP Validation and verification test suite
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
int _ompvv_isOffloadingOn = -1;

// Macro for output of information, warning and error messages
#ifdef VERBOSE_MODE
  #define OMPVV_WARNING(message, ...) { \
    printf("[OMPVV_WARNING: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
  }
  #define OMPVV_WARNING_IF(condition, message, ...) { \
    if(condition) { \
      printf("[OMPVV_WARNING: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
    } \
  }

  #define OMPVV_ERROR(message, ...) { \
    fprintf(stderr, "[OMPVV_ERROR: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
  }
  #define OMPVV_ERROR_IF(condition, message, ...) { \
    if(condition) { \
      fprintf(stderr, "[OMPVV_ERROR: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
    } \
  }

  #define OMPVV_INFOMSG(message, ...) { \
    printf("[OMPVV_INFO: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
  }
  #define OMPVV_INFOMSG_IF(condition, message, ...) { \
    if(condition) { \
      printf("[OMPVV_INFO: %s:%i] " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
    } \
  }
#else
  #define OMPVV_WARNING(message, ...) {}
  #define OMPVV_WARNING_IF(message, ...) {}
  #define OMPVV_ERROR(message, ...) {}
  #define OMPVV_ERROR_IF(message, ...) {}
  #define OMPVV_INFOMSG(message, ...) {}
  #define OMPVV_INFOMSG_IF(message, ...) {}
#endif // END IF VERBOSE_MODE

#define OMPVV_TEST_OFFLOADING_PROBE \
  _ompvv_isOffloadingOn = 0; \
_Pragma("omp target map (from: _ompvv_isOffloadingOn)") \
  {  _ompvv_isOffloadingOn = !omp_is_initial_device();  }

// Macro for checking if offloading is enabled or not
#define OMPVV_TEST_OFFLOADING { \
  OMPVV_TEST_OFFLOADING_PROBE \
  OMPVV_INFOMSG("Test is running on %s.",(_ompvv_isOffloadingOn)? "device" : "host"); \
}

// Macro for checking if offloading is enabled or not and set a variable with the result
#define OMPVV_TEST_AND_SET_OFFLOADING(var2set) { \
  OMPVV_TEST_OFFLOADING_PROBE \
  OMPVV_INFOMSG("Test is running on %s.",(_ompvv_isOffloadingOn)? "device" : "host"); \
  var2set = _ompvv_isOffloadingOn; \
}
// Macro for setting errors on condition
#define OMPVV_TEST_AND_SET(err, condition) { \
  int _ompvv_conditionEval = condition; \
  err += (_ompvv_conditionEval); \
}
// Macro for setting errors on condition and displaying an error if something went wrong
#define OMPVV_TEST_AND_SET_VERBOSE(err, condition) { \
  int _ompvv_conditionEval = condition; \
  err += (_ompvv_conditionEval); \
  OMPVV_ERROR_IF(_ompvv_conditionEval, " Condition " #condition " failed "); \
}
// Macro for reporting results
#define OMPVV_REPORT(err) { \
  OMPVV_INFOMSG("The value of " #err " is %d.", err); \
  if (_ompvv_isOffloadingOn == -1) \
    printf("[OMPVV_RESULT: %s] Test %s.\n", __FILENAME__, ((err) == 0)? "passed":"failed"); \
  else \
    printf("[OMPVV_RESULT: %s] Test %s on the %s.\n", __FILENAME__, ((err) == 0)? "passed":"failed", (_ompvv_isOffloadingOn)? "device" : "host"); \
}

// Macro for correct exit code
#define OMPVV_RETURN(err) { \
  return ((err) == 0) ? EXIT_SUCCESS : EXIT_FAILURE; \
}

// Macro for report and exit
#define OMPVV_REPORT_AND_RETURN(err) {\
  OMPVV_REPORT(err); \
  OMPVV_RETURN(err); \
}

// Macro to check if it is a shared data environment
#define OMPVV_TEST_SHARED_ENVIRONMENT_PROBE \
  int _ompvv_isSharedEnv = 0; \
  _ompvv_isOffloadingOn = 0; \
_Pragma("omp target map (from: _ompvv_isOffloadingOn) map(to: _ompvv_isSharedEnv)") \
  {  _ompvv_isOffloadingOn = !omp_is_initial_device();  \
     _ompvv_isSharedEnv = 1; \
  }

// Macro to report warning if it is a shared environment
#define OMPVV_TEST_SHARED_ENVIRONMENT {\
  OMPVV_TEST_SHARED_ENVIRONMENT_PROBE \
  OMPVV_WARNING_IF((_ompvv_isOffloadingOn && _ompvv_isSharedEnv == 1),"This tests is running on a shared data environment between host and device. This may cause errors") \
  }

// Macro to report warning if it is a shared environment and set a variable for further use
#define OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(var2set) {\
  OMPVV_TEST_SHARED_ENVIRONMENT_PROBE \
  OMPVV_WARNING_IF((_ompvv_isOffloadingOn && _ompvv_isSharedEnv == 1),"This tests is running on a shared data environment between host and device. This may cause errors") \
  var2set = (_ompvv_isOffloadingOn && _ompvv_isSharedEnv == 1);\
  }

// Macros to provide thread and team nums if they are not specified
#ifndef OMPVV_NUM_THREADS_DEVICE
  #define OMPVV_NUM_THREADS_DEVICE 8
#endif

#ifndef OMPVV_NUM_TEAMS_DEVICE
  #define OMPVV_NUM_TEAMS_DEVICE 8
#endif

#ifndef OMPVV_NUM_TEAMS_HOST
  #define OMPVV_NUM_TEAMS_HOST 4
#endif

#ifndef OMPVV_NUM_THREADS_HOST
  #define OMPVV_NUM_THREADS_HOST 8
#endif
