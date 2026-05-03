/*
This file is part of GOTCHA.  For copyright information see the COPYRIGHT
file in the top level directory, or at
https://github.com/LLNL/gotcha/blob/master/COPYRIGHT
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.  This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU Lesser General Public License
for more details.  You should have received a copy of the GNU Lesser General
Public License along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "tool3.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "gotcha/gotcha.h"
#include "gotcha/gotcha_types.h"

__attribute__((unused)) static char *retX_wrapper(char *x);

static gotcha_wrappee_handle_t origRetX_handle = 0x0;

#define NUM_IOFUNCS 1
struct gotcha_binding_t iofuncs3[] = {{"retX", retX_wrapper, &origRetX_handle}};

char *retX_wrapper(char *x) {
  typeof(&retX_wrapper) origRetX = gotcha_get_wrappee(origRetX_handle);
  strcat(x, "t3=>");

  return origRetX ? (origRetX(x)) : 0;
}

int init_tool3() {
  enum gotcha_error_t result;

  result = gotcha_wrap(iofuncs3, NUM_IOFUNCS, "tool3");
  if (result != GOTCHA_SUCCESS) {
    fprintf(stderr, "gotcha_wrap returned %d\n", (int)result);
    return -1;
  }

  return 0;
}
