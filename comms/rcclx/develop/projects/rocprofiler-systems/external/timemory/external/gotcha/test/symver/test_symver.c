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

#include <assert.h>
#include <gotcha/gotcha.h>
#include <stdio.h>

#include "retX.h"

// We need a place to store the pointer to the function we've wrapped
static gotcha_wrappee_handle_t origRetX_handle;

static int dogRetX(int x);

static struct gotcha_binding_t bindings[] = {
    {"retX", dogRetX, &origRetX_handle}};

// This is like a tool library's initialization function
static int wrap_init(void) {
  gotcha_wrap(bindings, 1, "symbol_version_test");
  return 0;
}

int dogRetX(int x) {
  typeof(&dogRetX) origRetX = gotcha_get_wrappee(origRetX_handle);
  printf("SO I FOR ONE THINK DOGS SHOULD RETURN %i\n", x);
  return origRetX ? origRetX(x) + 1 : 0;
}

int main(int ac, char *av[]) {
  wrap_init();
  int check_val = retX(9);
  assert(check_val == 10);
  return 0;
}
