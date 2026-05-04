/*
 * pfmlib_os_macos.c: set of functions for MacOS (Tiger)
 *
 * Copyright (c) 2008 Stephane Eranian
 * Contributed by Stephane Eranian <eranian@gmail.com>
 * As a sign of friendship to my friend Eric, big fan of MacOS
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysctl.h>

#include "pfmlib_priv.h"

typedef enum {
	TYPE_NONE,
	TYPE_STR,
	TYPE_INT
} mib_name_t;

/*
 * helper function to retrieve one value from /proc/cpuinfo
 * for internal libpfm use only
 * attr: the attribute (line) to look for
 * ret_buf: a buffer to store the value of the attribute (as a string)
 * maxlen : number of bytes of capacity in ret_buf
 *
 * ret_buf is null terminated.
 *
 * Return:
 * 	0 : attribute found, ret_buf populated
 * 	-1: attribute not found
 */
int
__pfm_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
	mib_name_t type = TYPE_NONE;
	union {
		char str[32];
		int val;
	} value;
	char *name = NULL;
	int mib[16];
	int ret = -1;
	size_t len, mib_len;

	if (attr == NULL || ret_buf == NULL || maxlen < 1)
		return -1;

	*ret_buf = '\0';

	if (!strcmp(attr, "vendor_id")) {
		name = 	"machdep.cpu.vendor";
		type = TYPE_STR;
	} else if (!strcmp(attr, "model")) {
		name = "machdep.cpu.model";
		type = TYPE_INT;
	} else if (!strcmp(attr, "cpu family")) {
		name = "machdep.cpu.family";
		type = TYPE_INT;
	}

	mib_len = 16;
	ret = sysctlnametomib(name, mib, &mib_len);
	if (ret)
		return -1;

	len = sizeof(value);
	ret = sysctl(mib, mib_len, &value, &len, NULL, 0);
	if (ret)
		return ret;

	if (type == TYPE_STR)
		strncpy(ret_buf, value.str, maxlen);
	else if (type == TYPE_INT)
		snprintf(ret_buf, maxlen, "%d", value.val);
	
	__pfm_vbprintf("attr=%s ret=%d ret_buf=%s\n", attr, ret, ret_buf);

	return ret;
}

void
pfm_init_syscalls(void)
{
}
