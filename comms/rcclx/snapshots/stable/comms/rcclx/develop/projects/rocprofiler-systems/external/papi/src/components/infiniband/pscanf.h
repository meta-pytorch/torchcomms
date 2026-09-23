/* This file was taken from the tacc_stats utility, which is distributed 
 * under a GPL license.
 */
#ifndef _PSCANF_H_
#define _PSCANF_H_
#include <stdio.h>
#include <stdarg.h>

__attribute__((format(scanf, 2, 3)))
  static inline int pscanf(const char *path, const char *fmt, ...)
{
  int rc = -1;
  FILE *file = NULL;
  char file_buf[4096];
  va_list arg_list;
  va_start(arg_list, fmt);

  file = fopen(path, "r");
  if (file == NULL)
    goto out;
  setvbuf(file, file_buf, _IOFBF, sizeof(file_buf));

  rc = vfscanf(file, fmt, arg_list);

 out:
  if (file != NULL)
    fclose(file);
  va_end(arg_list);
  return rc;
}

#endif
