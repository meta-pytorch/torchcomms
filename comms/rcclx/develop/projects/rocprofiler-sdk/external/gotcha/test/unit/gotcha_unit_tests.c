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

#include <check.h>
#include <fcntl.h>
#include <link.h>
#include <unistd.h>

#include "elf_ops.h"
#include "gotcha.c"
#include "gotcha/gotcha.h"
#include "gotcha_auxv.c"
#include "gotcha_auxv.h"
#include "gotcha_dl.h"
#include "gotcha_utils.h"
#include "hash.h"
#include "libc_wrappers.h"
#include "testing_lib.h"
#include "tool.c"
#include "tool.h"

#if !defined(STR)
#define STR(X) STR2(X)
#define STR2(X) #X
#endif
TCase *configured_case_create(const char *name) {
  TCase *ccase = tcase_create(name);
  tcase_set_timeout(ccase, 100.0);
  return ccase;
}

//////GOTCHA Core///Tests///////

tool_t *new_tool;
void setup_infrastructure() {
  if (!new_tool) new_tool = create_tool("internal_test_tool");
}

void teardown_infrastructure() {}

int check_pointer(void *ptr) { return ((ElfW(Addr))ptr) != 0; }

extern int gotcha_prepare_symbols(binding_t *bindings, int num_names);

int dummy_main(int argc, char *argv[]) {
  return argv[argc - 1][0];
}  // this is junk, will not be executed

gotcha_wrappee_handle_t orig_func_handle;
int wrap_sample_func() {
  typeof(&wrap_sample_func) orig_func = gotcha_get_wrappee(orig_func_handle);
  return orig_func() + 5;
}

START_TEST(auto_tool_creation) {
  gotcha_wrappee_handle_t my_main_handle;
  struct gotcha_binding_t bindings[] = {{"main", &dummy_main, &my_main_handle}};
  gotcha_wrap(bindings, 1, "sample_autocreate_tool");
  ck_assert_msg(1, "Should never fail unless segfault on tool creation");
  gotcha_wrap(bindings, 1, "internal_test_tool");
  ck_assert_msg(1, "Should never fail unless segfault on tool creation");
  gotcha_wrap(bindings, 1, "");
  remove_library(_r_debug.r_map->l_next);
  struct link_map *lib;
  for (lib = _r_debug.r_map; lib != 0; lib = lib->l_next) {
    remove_library(lib);
  }
}
END_TEST

START_TEST(symbol_wrap_test) {
  tool_t *tools = get_tool_list();
  ck_assert_msg((tools != NULL), "tools should exists");
  remove_tool_from_list(tools);
  tools = get_tool_list();
  ck_assert_msg((tools == NULL), "tools should not exists");
  struct gotcha_binding_t bindings[] = {
      {"simpleFunc", &wrap_sample_func, &orig_func_handle}};
  gotcha_wrap(bindings, 1, "internal_test_tool");
  int x = simpleFunc();
  ck_assert_msg((x != TESTING_LIB_RET_VAL),
                "gotcha_wrap did not redirect a call to the wrapper function");
  int priority;
  enum gotcha_error_t result =
      get_default_configuration_value(GOTCHA_PRIORITY, &priority);
  ck_assert_msg((result == GOTCHA_SUCCESS),
                "get_default_configuration_value should be successful");
  ck_assert_msg((priority == UNSET_PRIORITY), "priotity should not be set");
  result = get_configuration_value("dummy", GOTCHA_PRIORITY, &priority);
  ck_assert_msg((result == GOTCHA_INVALID_TOOL),
                "summy tool passed result should be GOTCHA_INVALID_TOOL");
  result = gotcha_configure_int("internal_test_tool", 2, priority);
  ck_assert_msg((result == GOTCHA_INTERNAL),
                "gotcha_configure_int passed should be GOTCHA_INTERNAL");
  result =
      get_configuration_value("internal_test_tool", GOTCHA_PRIORITY, &priority);
  ck_assert_msg(
      (result == GOTCHA_SUCCESS),
      "internal_test_tool tool passed result should be GOTCHA_SUCCESS");
  ck_assert_msg((priority == UNSET_PRIORITY), "priotity should not be set");
  result = gotcha_set_priority("internal_test_tool", 10);
  ck_assert_msg((result == GOTCHA_SUCCESS),
                "gotcha_set_priority should be GOTCHA_SUCCESS");
  result = gotcha_get_priority("internal_test_tool", &priority);
  ck_assert_msg((result == GOTCHA_SUCCESS),
                "gotcha_set_priority should be GOTCHA_SUCCESS");
  ck_assert_msg((priority == 10), "priority should be 10");
  priority = -1;
  result =
      get_configuration_value("internal_test_tool", GOTCHA_PRIORITY, &priority);
  ck_assert_msg(
      (result == GOTCHA_SUCCESS),
      "internal_test_tool tool passed result should be GOTCHA_SUCCESS");
  ck_assert_msg((priority == 10), "priotity should be set");
  result = get_configuration_value("internal_test_tool", 2, &priority);
  ck_assert_msg((result == GOTCHA_INTERNAL), "invalid property passed");
  struct binding_t *binds = get_bindings();
  ck_assert_msg((bindings != NULL), "should return all bindings");
  tools = get_tool_list();
  ck_assert_msg((tools != NULL), "tools should exists");
  binds = get_tool_bindings(tools);
  ck_assert_msg((bindings != NULL), "should return tool's bindings");
  gotcha_wrap(bindings, 1, "internal_test_tool2");
  ck_assert_msg((1), "should always pass");
  gotcha_wrap(bindings, 1, "internal_test_tool3");
  ck_assert_msg((1), "should always pass");
  tools = get_tool_list();
  ck_assert_msg((tools != NULL), "tools should exists");
  remove_tool_from_list(tools->next_tool->next_tool);
  tool_t *cur = get_tool_list(), *prev;
  while (cur) {
    prev = cur;
    cur = cur->next_tool;
    remove_tool_from_list(prev);
  }
  tools = get_tool_list();
  ck_assert_msg((tools == NULL), "tools should not exists");
}
END_TEST

START_TEST(bad_lookup_test) {
  struct gotcha_binding_t bindings[] = {
      {"this_is_the_story_of_a_function_we_shouldnt_find", &wrap_sample_func,
       &orig_func_handle}};
  enum gotcha_error_t errcode = gotcha_wrap(bindings, 1, "internal_test_tool");
  ck_assert_msg((errcode == GOTCHA_FUNCTION_NOT_FOUND),
                "Looked up a function that shouldn't be found and did not get "
                "correct error code");
  struct link_map *lib;
  for (lib = _r_debug.r_map; lib != 0; lib = lib->l_next) {
    if (gotcha_strstr(lib->l_name, "libinternal_unit_sample") != NULL) {
      break;
    }
  }
  ck_assert_msg((lib != NULL), "libinternal_unit_sample should be found");
  void *symbol;
  long result = lookup_exported_symbol("dummy", lib, &symbol);
  ck_assert_msg((result == -1), "Dummy symbol should not be found");

  result = lookup_exported_symbol("hiddenFunc", lib, &symbol);
  ck_assert_msg((result == -1),
                "hiddenFunc symbol should not be found as it is hidden");
}
END_TEST

Suite *gotcha_core_suite() {
  Suite *s = suite_create("Gotcha Core");
  TCase *core_case = configured_case_create("Wrapping");
  tcase_add_checked_fixture(core_case, setup_infrastructure,
                            teardown_infrastructure);
  tcase_add_test(core_case, symbol_wrap_test);
  tcase_add_test(core_case, bad_lookup_test);
  tcase_add_test(core_case, auto_tool_creation);
  suite_add_tcase(s, core_case);
  return s;
}

////////////Libc Wrapper///Tests////////////

START_TEST(gotcha_malloc_test) {
  int *x = (int *)gotcha_malloc(sizeof(int) * 10);
  x[9] = 5;
}
END_TEST

START_TEST(gotcha_free_test) {
  int *x = (int *)gotcha_malloc(sizeof(int) * 10);
  gotcha_free(x);
}
END_TEST

#define HEAP_TEST_SIZE 4096
static int check_heap_test(int **allocations) {
  int i, j;
  for (i = 0; i < HEAP_TEST_SIZE; i++) {
    if (!allocations[i]) continue;
    for (j = 0; j < i; j++) {
      if (allocations[i][j] != i + j) return -1;
    }
  }
  return 0;
}

START_TEST(gotcha_heap_test) {
  int i, j, k, l;
  int **allocations;

  // Do this test 10 times
  for (l = 0; l < 10; l++) {
    // Initialize allocations array to empty
    allocations = (int **)gotcha_malloc(sizeof(int *) * HEAP_TEST_SIZE);
    for (i = 0; i < HEAP_TEST_SIZE; i++) {
      allocations[i] = NULL;
    }
    ck_assert(check_heap_test(allocations) == 0);

    // Allocate each allocations[x] to be size x, and
    // allocations[x][y] to be x+y.  Do in sets of x%3.
    for (i = 0; i < 3; i++) {
      for (j = i; j < HEAP_TEST_SIZE; j += 3) {
        allocations[j] = (int *)gotcha_malloc(sizeof(int) * j);
        for (k = 0; k < j; k++) {
          allocations[j][k] = j + k;
        }
      }
      ck_assert(check_heap_test(allocations) == 0);
    }

    // Free allocations in sets of x%5
    for (i = 0; i < 5; i++) {
      for (j = i; j < HEAP_TEST_SIZE; j += 5) {
        gotcha_free(allocations[j]);
        allocations[j] = NULL;
      }
      ck_assert(check_heap_test(allocations) == 0);
    }
    gotcha_free(allocations);
  }
}
END_TEST

START_TEST(gotcha_realloc_test) {
  int *x = (int *)gotcha_malloc(sizeof(int) * 10);
  x[9] = 5;
  x = (int *)gotcha_realloc(x, sizeof(int) * 5);
  x[4] = 5;
  x = (int *)gotcha_realloc(x, sizeof(int) * 15);
  x[14] = 5;
  gotcha_free(x);
}
END_TEST

START_TEST(gotcha_memcpy_test) {
  int *x = (int *)gotcha_malloc(sizeof(int) * 10);
  int *y = (int *)gotcha_malloc(sizeof(int) * 10);
  int i = 0;
  for (i = 0; i < 10; i++) {
    x[i] = i;
    y[i] = 999;
  }
  gotcha_memcpy(y, x, sizeof(int) * 10);
  for (i = 0; i < 10; i++) {
    ck_assert_msg(y[i] == i,
                  "Target for gotcha_memcpy doesn't have the correct data");
  }
  gotcha_free(x);
  gotcha_free(y);
}
END_TEST

START_TEST(gotcha_strcmp_test) {
  ck_assert_msg(gotcha_strncmp("dogsaregood", "dogsisgreat", 4) == 0,
                "gotcha_strncmp is examining too many characters, marking "
                "matching prefixes as not matching");
  ck_assert_msg(gotcha_strncmp("dogsaregood", "dogsisgreat", 5) < 0,
                "gotcha_strncmp is examining too few characters, marking "
                "nonmatching prefixes as matching");
  ck_assert_msg(gotcha_strncmp("dogs", "pups", 999) < 0,
                "gotcha_strncmp fails on nonmatching strings of smaller "
                "lengths than the declared string length");
  ck_assert_msg(gotcha_strncmp("dogs", "dogs", 999) == 0,
                "gotcha_strncmp fails on matching strings of smaller lengths "
                "than the declared string length");
  ck_assert_msg(gotcha_strcmp("dogs", "pups") < 0,
                "gotcha_strcmp fails on nonmatching strings");
  ck_assert_msg(gotcha_strcmp("pups", "dogs") > 0,
                "gotcha_strcmp fails on reversed nonmatching strings");
  ck_assert_msg(gotcha_strcmp("dogs", "dogs") == 0,
                "gotcha_strcmp fails on matching strings");
  ck_assert_msg(gotcha_strstr("dogs", "og") != 0,
                "gotcha_strstr fails on matching strings");
  ck_assert_msg(gotcha_strstr("dogs", "gsa") == NULL,
                "gotcha_strstr fails on matching strings");
  ck_assert_msg(gotcha_strstr("dogs", "cats") == 0,
                "gotcha_strstr fails on nonmatching strings");
  ck_assert_msg(gotcha_strstr("dogs", "doges") == 0,
                "gotcha_strstr fails on nonmatching strings");
  ck_assert_msg(gotcha_strnlen("dogs", 10) == 4,
                "gotcha_strnlen works on strings");
  char v1[10] = "dogs";
  ck_assert_msg(gotcha_strncat(v1, "cats", 5) != NULL,
                "gotcha_strncat works on strings");
}
END_TEST

START_TEST(gotcha_atoi_test) {
  ck_assert_msg(gotcha_atoi("-82") == -82, "gotcha_atoi fails on -82");
  int x = gotcha_atoi("--82");
  // In libc, atoi returns 0 for this. 82 is arguably a good behavior
  ck_assert_msg((x == 82) || (x == 0), "gotcha_atoi fails on --82");
  ck_assert_msg(gotcha_atoi("99999993") == 99999993,
                "gotcha_atoi fails on 99999993");
}
END_TEST

// double_printf calls glibc printf to print a string to a buffer,
//  and calls gotcha_int_printf to print the same string to a fd
// The fd is the input to a pipe, and test_printf_buffers pulls
//  from the pipe and compares against the glibc buffer.
#define double_printf(FORMAT, ...)                                         \
  do {                                                                     \
    result_libc = snprintf(buffer, sizeof(buffer), FORMAT, ##__VA_ARGS__); \
    result_gotcha = gotcha_int_printf(pipe_out, FORMAT, ##__VA_ARGS__);    \
    test_printf_buffers(buffer, sizeof(buffer), pipe_out, pipe_in, FORMAT, \
                        result_libc, result_gotcha);                       \
  } while (0)

static void test_printf_buffers(char *buffer, int buffer_size, int pipe_out,
                                int pipe_in, const char *format,
                                int result_libc, int result_gotcha) {
  char pipe_buffer[4092];
  unsigned long i = 0, result;
  char zero = 0;

  buffer[buffer_size - 1] = '\0';
  gotcha_write(pipe_out, &zero, 1);

  if (result_libc != result_gotcha) {
    printf("\"%s\" printf test returned different result %d != %d\n", format,
           result_libc, result_gotcha);
    ck_assert_msg(result_libc == result_gotcha,
                  " printf test returned different results\n");
  }

  do {
    result = read(pipe_in, pipe_buffer + i, 1);
    ck_assert_msg(result == 1, "Failed to read from pipe buffer");
    i++;
  } while (i < sizeof(pipe_buffer) && pipe_buffer[i - 1] != '\0');

  if (gotcha_strcmp(buffer, pipe_buffer) != 0) {
    fprintf(stderr, "\"%s\" != \"%s\" for printf string \"%s\"\n", buffer,
            pipe_buffer, format);
  }
  ck_assert_msg(gotcha_strcmp(buffer, pipe_buffer) == 0,
                "Mismatch in printf test");
}

START_TEST(printf_test) {
  char buffer[4096], *format_str;
  int pipefds[2], pipe_in, pipe_out, result;
  int result_libc, result_gotcha;

  result = pipe2(pipefds, O_NONBLOCK);
  ck_assert_msg(result == 0, "Failed to create test pipe");
  pipe_in = pipefds[0];
  pipe_out = pipefds[1];

  double_printf("Hello, World!\n");

#define print_ints(SIGN, MULT, CODE, SIZET)                              \
  double_printf("%hh" CODE " is " STR(SIGN) " char\n",                   \
                (SIGN char)(10 * MULT));                                 \
  double_printf("%h" CODE " is " STR(SIGN) " short\n",                   \
                (SIGN short)(1000 * MULT));                              \
  double_printf("%" CODE " is " STR(SIGN) " int\n",                      \
                (SIGN int)(1000000 * MULT));                             \
  double_printf("%l" CODE " is " STR(SIGN) " long\n",                    \
                (SIGN long)(10000000000L * MULT));                       \
  double_printf("%ll" CODE " is " STR(SIGN) " long long\n",              \
                (SIGN long long)(1000000000L * MULT));                   \
  double_printf("%z" CODE " is " STR(SIZET) "\n", (SIZET)(1 * MULT));    \
  double_printf("All together: %hh" CODE " %h" CODE " %" CODE " %l" CODE \
                " %ll" CODE " %z" CODE "\n",                             \
                (SIGN char)MULT, (SIGN short)MULT, (SIGN int)MULT,       \
                (SIGN long)MULT, (SIGN long long)MULT, (SIZET)MULT)

  print_ints(signed, -5, "d", ssize_t);
  print_ints(signed, -1, "i", ssize_t);
  print_ints(signed, 3, "d", ssize_t);
  print_ints(signed, 2, "i", ssize_t);
  print_ints(signed, 0, "d", ssize_t);
  print_ints(unsigned, 1, "u", size_t);
  print_ints(unsigned, 2, "x", size_t);
  print_ints(unsigned, 3, "X", size_t);
  print_ints(unsigned, 0, "u", size_t);
  double_printf("%p is the address of buffer", buffer);

  double_printf("%s %s%s %s %s %d%s", "I", "am ", "a", "string", "printer.", 3,
                " is an int\n");
  double_printf("%s %s %s", "don't interpret ", "%s", " in argument\n");
  double_printf("Percent sign looks like %%%s\n", " this");

  double_printf("single string of characters: \a\b\f\r\t\v\'\"\?\\\n");
  double_printf("individual characters %c %c %c %c %c %c %c %c %c %c %c\n",
                '\a', '\b', '\f', '\r', '\t', '\v', '\n', '\'', '\"', '\?',
                '\\');
  double_printf(
      "don't interpret escaped chars: \\a\\b\\f\\r\\t\\v\\'\\\"\\?\\\\\\n");
  double_printf("%c%c%c%c%c%c", 'C', 'h', 'a', 'r', '%', '\n');

  format_str = "%y is not a valid code, but %s is.\n";
  double_printf(format_str, "%s");

  close(pipe_in);
  close(pipe_out);
}
END_TEST

Suite *gotcha_libc_suite() {
  Suite *s = suite_create("Gotcha Libc");
  TCase *libc_case = configured_case_create("Basic tests");
  tcase_add_test(libc_case, gotcha_malloc_test);
  tcase_add_test(libc_case, gotcha_free_test);
  tcase_add_test(libc_case, gotcha_realloc_test);
  tcase_add_test(libc_case, gotcha_heap_test);
  tcase_add_test(libc_case, gotcha_memcpy_test);
  tcase_add_test(libc_case, gotcha_strcmp_test);
  tcase_add_test(libc_case, gotcha_atoi_test);
  tcase_add_test(libc_case, printf_test);
  suite_add_tcase(s, libc_case);
  return s;
}

////////////GOTCHA Auxv///Tests////////////

START_TEST(vdso_map_test) {
  struct link_map *maps_vdso = get_vdso_from_maps();
  struct link_map *auxv_vdso = get_vdso_from_auxv();
  struct link_map *alias_vdso = get_vdso_from_aliases();
  int result = is_vdso(NULL);
  ck_assert_msg(result == 0, "VDSO should not be found");
  struct link_map *lib;
  for (lib = _r_debug.r_map; lib != 0; lib = lib->l_next) {
    if (gotcha_strstr(lib->l_name, "vdso") != NULL) {
      break;
    }
  }
  ck_assert_msg((lib != NULL), "vdso should be found");
  result = is_vdso(lib);
  ck_assert_msg(result == 1, "VDSO should be found");
  result = is_vdso(lib);
  ck_assert_msg(result == 1, "VDSO should be found");
  ck_assert_msg(maps_vdso || auxv_vdso || alias_vdso,
                "VDSO not found in any solution");
  ck_assert_msg(!maps_vdso || !auxv_vdso || maps_vdso == auxv_vdso,
                "VDSO Mismatch of maps and auxv");
  ck_assert_msg(!maps_vdso || !alias_vdso || maps_vdso == alias_vdso,
                "VDSO Mismatch of maps and alias");
  ck_assert_msg(!auxv_vdso || !alias_vdso || auxv_vdso == alias_vdso,
                "VDSO Mismatch of auxv and alias");
}
END_TEST

START_TEST(vdso_pagesize_test) {
  int vdso_pagesize = gotcha_getpagesize();
  ck_assert_msg(vdso_pagesize == getpagesize(),
                "VDSO does not contain page size");
}
END_TEST

START_TEST(auxv) {
  char *str = "A";
  unsigned long val;
  int result = read_hex(str, &val);
  ck_assert_msg(result == 1, "Incorrect length for hex");
  ck_assert_msg(val == 10, "Val should be 10");
}
END_TEST

Suite *gotcha_auxv_suite() {
  Suite *s = suite_create("Gotcha Auxv");
  TCase *libc_case = configured_case_create("Basic tests");
  tcase_add_test(libc_case, vdso_map_test);
  tcase_add_test(libc_case, vdso_pagesize_test);
  tcase_add_test(libc_case, auxv);
  suite_add_tcase(s, libc_case);
  return s;
}

////////////GOTCHA Hash///Tests////////////

#define NUM_INSERTS 16384

START_TEST(hash_grow_test) {
  hash_table_t table;
  int create_return_code = 0, add_return_code = 0, find_return_code = 0,
      remove_return_code = 0;
  create_return_code = create_hashtable(&table, 1, (hash_func_t)strhash,
                                        (hash_cmp_t)gotcha_strcmp);
  ck_assert_msg(!create_return_code, "Internal error creating hashtable");
  char *pointer_list[NUM_INSERTS];
  int loop;
  for (loop = 0; loop < NUM_INSERTS; loop++) {
    pointer_list[loop] = (char *)gotcha_malloc(sizeof(char) * 8);
    snprintf(pointer_list[loop], 8, "%d", loop);
    pointer_list[loop][7] = '\0';
  }
  for (loop = 0; loop < NUM_INSERTS; loop++) {
    add_return_code |=
        addto_hashtable(&table, pointer_list[loop], (void *)(long)loop);
  }
  ck_assert_msg(!add_return_code, "Internal error adding to hashtable");
  int first_broken = -1;
  for (loop = 0; loop < NUM_INSERTS; loop++) {
    int found_val;
    void *result;
    find_return_code |= lookup_hashtable(&table, pointer_list[loop], &result);
    found_val = (int)(long)result;
    if (found_val != loop) {
      first_broken = loop;
      break;
    }
  }
  ck_assert_msg(!find_return_code, "Internal error finding in hashtable");
  ck_assert_msg(first_broken == -1, "Failed to find item we searched for");
  for (loop = 0; loop < NUM_INSERTS; loop++) {
    remove_return_code |= removefrom_hashtable(&table, pointer_list[loop]);
  }
  ck_assert_msg(!remove_return_code, "Internal error removing from hashtable");
  int result = removefrom_hashtable(&table, pointer_list[0]);
  ck_assert_msg(result == -1, "all entries are removed should return -1");
  for (loop = 0; loop < NUM_INSERTS; loop++) {
    gotcha_free(pointer_list[loop]);
  }
}
END_TEST

Suite *gotcha_hash_suite() {
  Suite *s = suite_create("Gotcha Hashing");
  TCase *libc_case = configured_case_create("Basic tests");
  tcase_add_test(libc_case, hash_grow_test);
  suite_add_tcase(s, libc_case);
  return s;
}

START_TEST(gotcha_version_check) {
  ck_assert_msg(GOTCHA_GET_VERSION(1, 0, 3) > GOTCHA_GET_VERSION(1, 0, 2),
                "Check GOTCHA_GET_VERSION failed");
  ck_assert_msg(GOTCHA_VERSION >= GOTCHA_GET_VERSION(1, 0, 6),
                "Check GOTCHA_VERSION failed");
}
END_TEST

Suite *gotcha_version_suite() {
  Suite *s = suite_create("Gotcha Version");
  TCase *version_case = configured_case_create("Basic tests");
  tcase_add_test(version_case, gotcha_version_check);
  suite_add_tcase(s, version_case);
  return s;
}

////////////Launch///Tests////////////

int main() {
  int num_fails;
  Suite *core_suite = gotcha_core_suite();
  SRunner *core_runner = srunner_create(core_suite);
  srunner_run_all(core_runner, CK_NORMAL);
  num_fails = srunner_ntests_failed(core_runner);
  Suite *libc_suite = gotcha_libc_suite();
  SRunner *libc_runner = srunner_create(libc_suite);
  srunner_run_all(libc_runner, CK_NORMAL);
  num_fails += srunner_ntests_failed(libc_runner);
  Suite *auxv_suite = gotcha_auxv_suite();
  SRunner *auxv_runner = srunner_create(auxv_suite);
  srunner_run_all(auxv_runner, CK_NORMAL);
  num_fails += srunner_ntests_failed(auxv_runner);
  Suite *hash_suite = gotcha_hash_suite();
  SRunner *hash_runner = srunner_create(hash_suite);
  srunner_run_all(hash_runner, CK_NORMAL);
  num_fails += srunner_ntests_failed(hash_runner);
  Suite *version_suite = gotcha_version_suite();
  SRunner *version_runner = srunner_create(version_suite);
  srunner_run_all(version_runner, CK_NORMAL);
  num_fails += srunner_ntests_failed(version_runner);
  srunner_free(core_runner);
  srunner_free(libc_runner);
  srunner_free(auxv_runner);
  srunner_free(hash_runner);
  srunner_free(version_runner);
  return num_fails;
}
