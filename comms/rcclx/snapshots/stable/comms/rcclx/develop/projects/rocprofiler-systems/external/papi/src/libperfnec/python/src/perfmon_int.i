/*

 * Copyright (c) 2008 Google, Inc.
 * Contributed by Arun Sharma <arun.sharma@google.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * Python Bindings for perfmon.
 */
%module perfmon_int
%{
#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

static PyObject *libpfm_err;
%}

%include "carrays.i"
%include "cstring.i"
%include <stdint.i>

/* Some typemaps for corner cases SWIG can't handle */
/* Convert from Python --> C */
%typemap(memberin) pfmlib_event_t[ANY] {
  int i;
  for (i = 0; i < $1_dim0; i++) {
      $1[i] = $input[i];
  }
}

%typemap(out) pfmlib_event_t[ANY] {  
  int len, i;
  len = $1_dim0;
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyObject *o = SWIG_NewPointerObj(SWIG_as_voidptr(&$1[i]),
                                     SWIGTYPE_p_pfmlib_event_t, 0 |  0 );
    PyList_SetItem($result, i, o);
  }
}

/* Convert from Python --> C */
%typemap(memberin) pfmlib_reg_t[ANY] {
  int i;
  for (i = 0; i < $1_dim0; i++) {
      $1[i] = $input[i];
  }
}

%typemap(out) pfmlib_reg_t[ANY] {  
  int len, i;
  len = $1_dim0;
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyObject *o = SWIG_NewPointerObj(SWIG_as_voidptr(&$1[i]),
                                     SWIGTYPE_p_pfmlib_reg_t, 0 |  0 );
    PyList_SetItem($result, i, o);
  }
}

/* Convert libpfm errors into exceptions */
%typemap(out) os_err_t {  
  if (result == -1) {
    PyErr_SetFromErrno(PyExc_OSError);
    SWIG_fail;
  } 
  resultobj = SWIG_From_int((int)(result));
};

%typemap(out) pfm_err_t {  
  if (result != PFMLIB_SUCCESS) {
    PyObject *obj = Py_BuildValue("(i,s)", result, 
                                  pfm_strerror(result));
    PyErr_SetObject(libpfm_err, obj);
    SWIG_fail;
  } else {
    PyErr_Clear();
  }
  resultobj = SWIG_From_int((int)(result));
}

/* Convert libpfm errors into exceptions */
%typemap(out) os_err_t {  
  if (result == -1) {
    PyErr_SetFromErrno(PyExc_OSError);
    SWIG_fail;
  } 
  resultobj = SWIG_From_int((int)(result));
};

%typemap(out) pfm_err_t {  
  if (result != PFMLIB_SUCCESS) {
    PyObject *obj = Py_BuildValue("(i,s)", result, 
                                  pfm_strerror(result));
    PyErr_SetObject(libpfm_err, obj);
    SWIG_fail;
  } else {
    PyErr_Clear();
  }
  resultobj = SWIG_From_int((int)(result));
}

%cstring_output_maxsize(char *name, size_t maxlen)
%cstring_output_maxsize(char *name, int maxlen)

%extend pfmlib_regmask_t {
  unsigned int weight() {
    unsigned int w = 0;
    pfm_regmask_weight($self, &w);
    return w;
  }
}

/* Kernel interface */
%include <perfmon/perfmon.h>
%array_class(pfarg_pmc_t, pmc)
%array_class(pfarg_pmd_t, pmd)

/* Library interface */
%include <perfmon/pfmlib.h>

%extend pfarg_ctx_t {
  void zero() {
    memset(self, 0, sizeof(self));
  }
}

%extend pfarg_load_t {
  void zero() {
    memset(self, 0, sizeof(self));
  }
}


%init %{
  libpfm_err = PyErr_NewException("perfmon.libpfmError", NULL, NULL);
  PyDict_SetItemString(d, "libpfmError", libpfm_err); 
%}

%inline %{
/* Helper functions to avoid pointer classes */
int pfm_py_get_pmu_type(void) {
  int tmp = -1;
  pfm_get_pmu_type(&tmp);
  return tmp;
}

unsigned int pfm_py_get_hw_counter_width(void) {
  unsigned int tmp = 0;
  pfm_get_hw_counter_width(&tmp);
  return tmp;
}

unsigned int pfm_py_get_num_events(void) {
  unsigned int tmp = 0;
  pfm_get_num_events(&tmp);
  return tmp;
}

int pfm_py_get_event_code(int idx) {
  int tmp = 0;
  pfm_get_event_code(idx, &tmp);
  return tmp;
}

unsigned int pfm_py_get_num_event_masks(int idx) {
  unsigned int tmp = 0;
  pfm_get_num_event_masks(idx, &tmp);
  return tmp;
}

unsigned int pfm_py_get_event_mask_code(int idx, int i) {
  unsigned int tmp = 0;
  pfm_get_event_mask_code(idx, i, &tmp);
  return tmp;
}

#define PFMON_MAX_EVTNAME_LEN   128

PyObject *pfm_py_get_event_name(int idx) {
  char name[PFMON_MAX_EVTNAME_LEN];

  pfm_get_event_name(idx, name, PFMON_MAX_EVTNAME_LEN);
  return PyString_FromString(name);
}

PyObject *pfm_py_get_event_mask_name(int idx, int i) {
  char name[PFMON_MAX_EVTNAME_LEN];

  pfm_get_event_mask_name(idx, i, name, PFMON_MAX_EVTNAME_LEN);
  return PyString_FromString(name);
}

PyObject *pfm_py_get_event_description(int idx) {
  char *desc;
  PyObject *ret;
  pfm_get_event_description(idx, &desc);
  ret = PyString_FromString(desc);
  free(desc);
  return ret;
}

PyObject *pfm_py_get_event_mask_description(int idx, int i) {
  char *desc;
  PyObject *ret;
  pfm_get_event_mask_description(idx, i, &desc);
  ret = PyString_FromString(desc);
  free(desc);
  return ret;
}

%}
