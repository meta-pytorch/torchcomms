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
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

int foo(void);
int bar(void);

static void* impl_lib;
static int (*impl_foo)(void);
static int (*impl_bar)(void);

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
void dispatch_init(void) {
  fprintf(stderr, "Ed dispatch_init()\n");

  impl_lib = dlopen("libimpl.so", RTLD_NOW);
  assert(impl_lib);
  impl_foo = dlsym(impl_lib, "foo");
  assert(impl_foo);
  impl_bar = dlsym(impl_lib, "bar");
  assert(impl_bar);

  int ret = impl_bar();

  fprintf(stderr, "Ld dispatch_init() = %d\n", ret);
}

int foo(void) {
  fprintf(stderr, "Ed foo()\n");

  pthread_once(&init_once, dispatch_init);

  int ret = impl_bar() + impl_foo();

  fprintf(stderr, "Ld foo()\n");

  return ret;
}

int bar(void) {
  fprintf(stderr, "Ed bar()\n");

  pthread_once(&init_once, dispatch_init);

  int ret = impl_bar();

  fprintf(stderr, "Ld bar()\n");

  return ret;
}
