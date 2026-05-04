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

#define _GNU_SOURCE
#include <gotcha/gotcha.h>
#include <stdio.h>

int foo(void);
int bar(void);

static gotcha_wrappee_handle_t handle_foo;
static gotcha_wrappee_handle_t handle_bar;

static int do_foo(void) {
  fprintf(stderr, "Ew foo()\n");

  typeof(&do_foo) orig_foo = gotcha_get_wrappee(handle_foo);
  int ret = orig_foo();

  fprintf(stderr, "Lw foo() = %d\n", ret);

  return ret;
}

static int do_bar(void) {
  fprintf(stderr, "Ew bar()\n");

  typeof(&do_bar) orig_bar = gotcha_get_wrappee(handle_bar);
  int ret = orig_bar();

  fprintf(stderr, "Lw bar() = %d\n", ret);

  return ret;
}

static struct gotcha_binding_t bindings[] = {
    {"foo", do_foo, &handle_foo},
    {"bar", do_bar, &handle_bar},
};

int main(int ac, char *av[]) {
  gotcha_wrap(bindings, 2, "test");
  printf("%d\n", foo());

  return 0;
}
