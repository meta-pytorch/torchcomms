/*
 * libpfms.h - header file for libpfms - a helper library for perfmon SMP monitoring
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
#ifndef __LIBPFMS_H__
#define __LIBPFMS_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*pfms_ovfl_t)(pfarg_msg_t *msg);

int pfms_initialize(void);
int pfms_create(uint64_t *cpu_list, size_t n, pfarg_ctx_t *ctx, pfms_ovfl_t *ovfl, void **desc);
int pfms_write_pmcs(void *desc, pfarg_pmc_t *pmcs, uint32_t n);
int pfms_write_pmds(void *desc, pfarg_pmd_t *pmds, uint32_t n);
int pfms_read_pmds(void *desc, pfarg_pmd_t *pmds, uint32_t n);
int pfms_start(void *desc);
int pfms_stop(void *desc);
int pfms_close(void *desc);
int pfms_unload(void *desc);
int pfms_load(void *desc);

#ifdef __cplusplus /* extern C */
}
#endif
#endif /* __LIBPFMS_H__ */
