#ifndef _CAT_PARAMS_
#define _CAT_PARAMS_

typedef struct cat_params_s{
    int subsetsize;
    int mode;
    int max_iter;
    int bench_type;
    int show_progress;
    int quick;
    char *conf_file;
    char *inputfile;
    char *outputdir;
} cat_params_t;

#endif // _CAT_PARAMS_
