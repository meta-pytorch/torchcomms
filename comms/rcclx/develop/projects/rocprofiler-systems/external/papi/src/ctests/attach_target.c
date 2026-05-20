#include <stdio.h>
#include <stdlib.h>

#include "do_loops.h"


int main(int argc, char **argv)
{
	int c, i = NUM_FLOPS;

	if (argc > 1) {
		c =  atoi(argv[1]);
		if (c >= 0) {
			i = c;
		}
	}

	do_flops(i);

	return 0;
}
