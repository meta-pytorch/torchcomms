#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void simple_init(void);
double simple_compute(double x);

int main(int argc, char **argv){
    int i;
    int be_verbose = 0;

    if( (argc > 1) && !strcmp(argv[1], "-verbose") )
        be_verbose = 1;

    simple_init();

    for(i=0; i<10; i++){
        double sum;

        sum = simple_compute(0.87*i);
        if( be_verbose) printf("sum=%lf\n",sum);
    }

    // This test exists just to check that a code that links against libsde
    // _without_ linking against libpapi will still compile and run. Therefore,
    // if we got to this point then the test has passed.
    fprintf( stdout, "%sPASSED%s\n","\033[1;32m","\033[0m");

    return 0;
}
