
/****************************/                                                                                                     
/* THIS IS OPEN SOURCE CODE */                                                                                                     
/****************************/                                                                                                     
                                                                                                                                   
/**                                                                                                                                
  * @file:    PAPI_Matlab.c   
  * @author   Frank Winkler <frank.winkler@icl.utk.edu>
  *
  *	@brief PAPI Matlab integration.
  *	See PAPI_Matlab.readme for more information.
  */

#define FLIPS_EVENT PAPI_FP_INS
#define FLOPS_EVENT PAPI_FP_OPS

#include "mex.h"
#include "matrix.h"
#include "papi.h"

static long long accum_error = 0;
static long long start_time = 0;
int EventSet = PAPI_NULL;
int papi_init = 0;
int papi_start = 0;

void initialize_papi() {
  int result;

  /* initialize PAPI */
  result = PAPI_library_init(PAPI_VER_CURRENT);
  if(result < PAPI_OK) {
    mexPrintf("Error code: %d\n", result);
    mexErrMsgTxt("Error PAPI_create_eventset.");
  }

  /* create EventSet */
  result = PAPI_create_eventset(&EventSet);
  if(result < PAPI_OK) {
    mexPrintf("Error code: %d\n", result);
    mexErrMsgTxt("Error PAPI_library_init.");
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
  float real_time, proc_time, rate;
  int i;
  int number_of_counters;
  unsigned int mrows, nchars;
  unsigned int *events;
  unsigned int flop_events[2];
  long long ins = 0, *values, flop_values[2];
  long long elapsed_time;
  int result;
  char *input, *temp;
  char one_output[]	= "This function produces one output per running counter.";
  char no_input[]	= "This function expects no input.";
  char error_reading[]	= "Error reading the running counters.";

  if ( papi_init == 0 ) {
    initialize_papi();
    papi_init = 1;
  }

  /* Check for proper number of arguments. */
  if(nrhs < 1) {
    mexErrMsgTxt("This function expects input.");
  }
  nchars = mxGetNumberOfElements(prhs[0]);
  input = (char *)mxCalloc(nchars, sizeof(char) + 1);
  input = mxArrayToString(prhs[0]);

  if(!strncmp(input, "num", 3)) {
    if(nrhs != 1) {
      mexErrMsgTxt(no_input);
    }
    else if(nlhs != 1) {
      mexErrMsgTxt("This function produces one and only one output: counters.");
    }
    result = PAPI_num_cmp_hwctrs(0);
    if(result < PAPI_OK) {
      mexPrintf("Error code: %d\n", result);
      mexErrMsgTxt("Error reading counters.");
    }
    plhs[0] = mxCreateDoubleScalar((double)result);
  }

  else if((!strncmp(input, "flip", 4)) || (!strncmp(input, "flop", 4))) {
    if(nrhs != 1) {
      mexErrMsgTxt(no_input);
    } else if(nlhs > 2) {
      if (input[2] == 'i')
        mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, mflips].");
      else
        mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, mflops].");
    }
    if (input[2] == 'i') {
      if(result = PAPI_flips_rate( FLIPS_EVENT, &real_time, &proc_time, &ins, &rate)<PAPI_OK) {
        mexPrintf("Error code: %d\n", result);
        mexErrMsgTxt("Error getting flips.");
      }
    } else {
       if(result = PAPI_event_name_to_code("EMON_SSE_SSE2_COMP_INST_RETIRED_PACKED_DOUBLE", &(flop_events[0])) < PAPI_OK) {
          if(result = PAPI_flops_rate( FLOPS_EVENT, &real_time, &proc_time, &ins, &rate)<PAPI_OK) {
            mexPrintf("Error code: %d\n", result);
            mexErrMsgTxt("Error getting flops.");
          }
       } else {
         if(start_time == 0) {
            flop_events[1] = PAPI_FP_OPS;
            start_time = PAPI_get_real_usec();

            for (i = 0; i < 2; i++) {
              result = PAPI_add_event(EventSet, flop_events[i]);
              if(result < PAPI_OK) {
                mexPrintf("Error code: %d\n", result);
                mexErrMsgTxt("Error PAPI_create_eventset.");
              }
            }

            if((result = PAPI_start(EventSet)) < PAPI_OK) {
               mexPrintf("Error code: %d\n", result);
               mexErrMsgTxt("Error getting flops.");
            } else {
               papi_start = 1;
               ins = 0;
               rate = 0;
            }
         } else {
            if((result = PAPI_read(EventSet, values)) < PAPI_OK) {
               mexPrintf("%d\n", result);
               mexErrMsgTxt(error_reading);
            } else {
               elapsed_time = PAPI_get_real_usec() - start_time;
               ins = (2*flop_values[0])+flop_values[1];
               rate = ((float)ins)/((float)elapsed_time);
            }
         }
       }
    }
    if(nlhs > 0) {
     plhs[0] = mxCreateDoubleScalar((double)(ins - accum_error));
      /* this call adds 7 fp instructions to the total */
      /* but apparently not on Pentium M with Matlab 7.0.4 */
/*      accum_error += 7; */
      if(nlhs == 2) {
        plhs[1] = mxCreateDoubleScalar((double)rate);
        /* the second call adds 4 fp instructions to the total */
      /* but apparently not on Pentium M with Matlab 7.0.4 */
/*        accum_error += 4; */
      }
    }
  }

  else if(!strncmp(input, "start", 5)) {
    if(nlhs != 0) {
      mexErrMsgTxt("This function produces no output.");
    }
    if(nrhs > (PAPI_num_cmp_hwctrs(0) + 1)) {
      mexErrMsgTxt(one_output);
    }
    mrows = mxGetM(prhs[1]);
    events = (unsigned int *)mxCalloc(nrhs - 1, sizeof(int) + 1);
    for(i = 1; i < nrhs; i++) {
      if(mxIsComplex(prhs[i]) || !(mrows == 1) ) {
        mexErrMsgTxt("Input must be a list of strings.");
      }
      if(mxIsChar(prhs[i])) {
        nchars = mxGetNumberOfElements(prhs[i]);
        temp = (char *)mxCalloc(nchars, sizeof(char) + 1);
        temp = mxArrayToString(prhs[i]);
        if(result = PAPI_event_name_to_code(temp, &(events[i - 1])) < PAPI_OK) {
          mxFree(temp);
          mexPrintf("Error code: %d\n", result);
          mexErrMsgTxt("Incorrect PAPI code given.");
       }
        mxFree(temp);
      }
      else {
        events[i - 1] = (unsigned int)mxGetScalar(prhs[i]);
      }
    }

    if((result = PAPI_cleanup_eventset(EventSet)) < PAPI_OK)
      mexErrMsgTxt("Error PAPI_cleanup_eventset");

    for (i = 0; i < nrhs - 1; i++) {
      result = PAPI_add_event(EventSet, events[i]);
      if(result < PAPI_OK) {
        mexPrintf("Error code: %d\n", result);
        mexErrMsgTxt("Error PAPI_add_event.");
      }
    }
    if((result = PAPI_start(EventSet)) < PAPI_OK) {
      mxFree(events);
      mexPrintf("Error code: %d\n", result);
      mexErrMsgTxt("Error initializing counters.");
    }
    papi_start = 1;
    mxFree(events);
  }

  else if(!strncmp(input, "stop", 4)) {
    if(nrhs != 1) {
      mexErrMsgTxt(no_input);
    }
    number_of_counters = PAPI_num_cmp_hwctrs(0);
    if(nlhs > number_of_counters ) {
      mexErrMsgTxt(one_output);
    }
  if (nlhs == 0) 
    values = (long long*)mxCalloc(number_of_counters, sizeof(long long));
  else 
    values = (long long *)mxCalloc(nlhs, sizeof(long long) + 1);

  result = PAPI_OK;
  if (start_time == 0) {
    if ( papi_start == 1 )
      result = PAPI_stop(EventSet, values);
  } else {
    start_time = 0;
    if ( papi_start == 1 )
      result = PAPI_stop(EventSet, flop_values);
  }
  PAPI_rate_stop();
  papi_start = 0;

  if(result < PAPI_OK) {
    if(result != PAPI_ENOTRUN) {
      mexPrintf("Error code: %d\n", result);
      mexErrMsgTxt("Error stopping the running counters.");
    }
  }
  accum_error = 0;
  for(i = 0; i < nlhs; i++) {
    plhs[i] = mxCreateDoubleScalar((double)values[i]);
  }
  mxFree(values);
  }

  else if(!strncmp(input, "read", 4)) {
    if(nrhs != 1) {
      mexErrMsgTxt(no_input);
    }
    if(nlhs > PAPI_num_cmp_hwctrs(0)) {
      mexErrMsgTxt(one_output);
    }
    values = (long long *)mxCalloc(nlhs, sizeof(long long) + 1);
    if((result = PAPI_read(EventSet, values)) < PAPI_OK) {
      mexPrintf("%d\n", result);
      mexErrMsgTxt(error_reading);
    }
    for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateDoubleScalar((double)values[i]);
    }
    mxFree(values);
  }

  else if(!strncmp(input, "accum", 5)) {
    if(nrhs > PAPI_num_cmp_hwctrs(0) + 1) {
      mexErrMsgTxt(no_input);
    }
    if(nlhs > PAPI_num_cmp_hwctrs(0)) {
      mexErrMsgTxt(one_output);
    }
    values = (long long *)mxCalloc(nlhs, sizeof(long long) + 1);
    for(i = 0; i < nrhs - 1; i++) {
      values[i] = (long long)(*(mxGetPr(prhs[i + 1])));
    }
    if(result = PAPI_accum(EventSet, values) < PAPI_OK) {
      mexPrintf("Error code: %d\n", result);
      mexErrMsgTxt(error_reading);
    }
    for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateDoubleScalar((double)values[i]);
    }
    mxFree(values);
  }

  else if(!strncmp(input, "ipc", 3)) {
    if(nrhs != 1) {
      mexErrMsgTxt(no_input);
    } else if(nlhs > 2) {
      mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, ipc].");
    }
    if(PAPI_ipc(&real_time, &proc_time, &ins, &rate)<PAPI_OK) {
      mexErrMsgTxt("Error getting instruction rate.");
    }
    if(nlhs > 0) {
      plhs[0] = mxCreateDoubleScalar((double)ins);
      if(nlhs == 2) {
        plhs[1] = mxCreateDoubleScalar((double)rate);
      }
    }
  }

  else {
    mexPrintf("Cannot find the command you specified.\n");
    mexErrMsgTxt("See the included readme file.");
  }
}
