//
// Created by haridev on 7/27/23.
//
#include <gotcha/gotcha.h>
#include <stdio.h>

#include "num.h"
int correct_return_four() { return 4; }

static gotcha_wrappee_handle_t buggy_return_four;
struct gotcha_binding_t funcs[] = {
    {"return_four", correct_return_four, &buggy_return_four}};
int main() {
  int (*fp)(void) = &return_four;
  int result = fp();
  if (result != 3) {
    fprintf(stderr, "the libnum.so would return 3\n");
    return -1;
  }
  result = gotcha_wrap(funcs, 1, "dlopen_test");
  if (result == GOTCHA_FUNCTION_NOT_FOUND) {
    fprintf(stderr, "GOTCHA should have found a function, but found it\n");
    return -1;
  }
  int (*fp2)(void) = &return_four;
  result = fp2();
  if (result != 4) {
    fprintf(stderr, "wrapper should return 4\n");
    return -1;
  }
  return 0;
}