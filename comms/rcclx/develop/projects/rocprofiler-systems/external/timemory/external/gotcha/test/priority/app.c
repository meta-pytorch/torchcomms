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

#include <gotcha/gotcha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib1.h"
#include "tool1.h"
#include "tool2.h"
#include "tool3.h"
extern int init_autotee(char *filename);
extern int close_autotee();

#define OUTPUT_FILE "tee.out"
char *get_string(char *in) {
  if (in) {
    free(in);
  }
  char *str = (char *)malloc(sizeof(char) * 15);
  memset(str, 0, sizeof(const char) * 15);
  return str;
}
#define Q(x) #x
#define QUOTE(x) Q(x)

void bubble3(int *in) {
  int indices[3];
  indices[0] = 1;
  indices[1] = 2;
  indices[2] = 3;
  int i = 0;
  int k = 0;
  for (i = 0; i < 3; i++) {
    for (k = 0; k < 2; k++) {
      if (in[k] <= in[k + 1]) {
        int temp = in[k];
        in[k] = in[k + 1];
        in[k + 1] = temp;
        temp = indices[k];
        indices[k] = indices[k + 1];
        indices[k + 1] = temp;
      }
    }
  }
  in[0] = indices[0];
  in[1] = indices[1];
  in[2] = indices[2];
}
int priority_from_str(const char *in, int default_val) {
  return in ? atoi(in) : default_val;
}
char *get_expected_result(int p1, int p2, int p3) {
  char *x = (char *)malloc(sizeof(char) * 13);
  int ordering[3];
  ordering[0] = p1;
  ordering[1] = p2;
  ordering[2] = p3;
  bubble3(ordering);
  p1 = ordering[0];
  p2 = ordering[1];
  p3 = ordering[2];
  sprintf(x, "t%d=>t%d=>t%d=>", p1, p2, p3);
  return x;
}
int main() {
  const char *env_one = getenv("tool_one_priority");
  const char *env_two = getenv("tool_two_priority");
  const char *env_three = getenv("tool_three_priority");
  int tool_one_priority = priority_from_str(env_one, 1);
  int tool_two_priority = priority_from_str(env_two, 2);
  int tool_three_priority = priority_from_str(env_three, 3);

  char *expected_result = get_expected_result(
      tool_one_priority, tool_two_priority, tool_three_priority);
  int result;
  // breaks on:: 321
  char *str = NULL;
  gotcha_set_priority("tool1", tool_one_priority);
  gotcha_set_priority("tool2", tool_two_priority);
  gotcha_set_priority("tool3", tool_three_priority);
  result = init_tool1();
  result = init_tool2();
  result = init_tool3();
  if (result != 0) return -1;
  str = get_string(str);
  retX(str);
  int fail = strncmp(str, expected_result, 12);
  if (fail) {
    fprintf(stderr, "Failed test, expected %s, got %s\n", expected_result, str);
  }
  return fail;
}
