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
#include <dlfcn.h>
#include <stdio.h>

#include "gotcha/gotcha.h"
#include "num.h"

int wrong_return_four() { return 5; }
int wrong_return_five() { return 4; }
int wrong_return_six() { return 3; }
static gotcha_wrappee_handle_t buggy_return_four;
static gotcha_wrappee_handle_t buggy_return_five;
static gotcha_wrappee_handle_t buggy_return_six;
struct gotcha_binding_t func_four[] = {
    {"return_four", wrong_return_four, &buggy_return_four}};
struct gotcha_binding_t func_five[] = {
    {"return_five", wrong_return_five, &buggy_return_five}};
struct gotcha_binding_t func_six[] = {
    {"return_six", wrong_return_six, &buggy_return_six}};

int main() {
  int result;
  int had_error = 0;
  result = gotcha_wrap(func_four, 1, "test_filter_four");
  if (result == GOTCHA_FUNCTION_NOT_FOUND) {
    fprintf(stderr, "GOTCHA should find a function, but found it\n");
    return -1;
  }
  result = return_four();
  if (result != 5) {
    fprintf(stderr, "ERROR: wrapper function should return 5\n");
    had_error = -1;
  }
  gotcha_filter_libraries_by_name("libnum3.so");
  result = gotcha_wrap(func_five, 1, "test_filter_five");
  if (result == GOTCHA_FUNCTION_NOT_FOUND) {
    fprintf(stderr, "GOTCHA should find a function, but found it\n");
    return -1;
  }
  result = return_five();
  if (result != 5) {
    fprintf(stderr,
            "ERROR: library function should return 5. no wrapping for exec\n");
    had_error = -1;
  }
  gotcha_only_filter_last();
  result = gotcha_wrap(func_six, 1, "test_filter_six");
  if (result == GOTCHA_FUNCTION_NOT_FOUND) {
    fprintf(stderr, "GOTCHA should find a function, but found it\n");
    return -1;
  }
  result = return_six();
  if (result != 6) {
    fprintf(stderr,
            "ERROR: library function should return 6 no wrapping for exec\n");
    had_error = -1;
  }
  gotcha_restore_library_filter_func();
  gotcha_wrap(func_six, 1, NULL);
  return had_error;
}
