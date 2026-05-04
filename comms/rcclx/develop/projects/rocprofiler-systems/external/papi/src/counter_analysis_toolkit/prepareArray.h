#ifndef _PREPARE_ARRAY_
#define _PREPARE_ARRAY_

#include <stdint.h>

#define RANDOM 0x2
#define SECRND 0x3
#define SEQUEN 0x4

int prepareArray(uintptr_t *array, long long len, long long stride, long long secSize, int pattern);

#endif
