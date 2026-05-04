int compar_lf(const void *a, const void *b){
    const double *da = (const double *)a;
    const double *db = (const double *)b;
    if( *da < *db) return -1;
    if( *da > *db) return 1;
    return 0;
}

int compar_lld(const void *a, const void *b){
    const long long int *da = (const long long int *)a;
    const long long int *db = (const long long int *)b;
    if( *da < *db) return -1;
    if( *da > *db) return 1;
    return 0;
}
