#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "driver.h"

#if defined(USE_MPI)
#include <mpi.h>
#endif

int main(int argc, char*argv[])
{
    int cmbtotal = 0, ct = 0, track = 0, ret = 0;
    int i, nevts = 0, status;
    int *cards = NULL, *indexmemo = NULL;
    char **allevts = NULL, **basenames = NULL;
    evstock *data = NULL;
    cat_params_t params = {-1,0,1,0,0,0,NULL,NULL,NULL};
    int nprocs = 1, myid = 0;

#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif

    // Initialize PAPI.
    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if(ret != PAPI_VER_CURRENT){

        fprintf(stderr,"PAPI shared library version error: %s Exiting...\n", PAPI_strerror(ret));
        return 0;
    }

    // Initialize PAPI thread support.
    ret = PAPI_thread_init( omp_get_thread_num_wrapper );
    if( ret != PAPI_OK ) {

        fprintf(stderr,"PAPI thread init error: %s Exiting...\n", PAPI_strerror(ret));
        return 0;
    }

    // Parse the command-line arguments.
    status = parseArgs(argc, argv, &params);
    if(0 != status)
    {
        free(params.outputdir);
        PAPI_shutdown();
        return 0;
    }

    // Allocate space for the native events and qualifiers.
    data = (evstock*)calloc(1,sizeof(evstock));
    if(NULL == data)
    {
        free(params.outputdir);
        fprintf(stderr, "Could not initialize event stock. Exiting...\n");
        PAPI_shutdown();
        return 0;
    }

    // Read the list of base event names and maximum qualifier set cardinalities.
    if( READ_FROM_FILE == params.mode)
    {
        ct = setup_evts(params.inputfile, &basenames, &cards);
        if(ct == -1)
        {
            free(params.outputdir);
            remove_stock(data);
            PAPI_shutdown();
            return 0;
        }
    }

    // Populate the event stock.
    status = build_stock(data);
    if(status)
    {
        free(params.outputdir);
        if(READ_FROM_FILE == params.mode)
        {
            for(i = 0; i < ct; ++i)
            {
                free(basenames[i]);
            }
            free(basenames);
            free(cards);
        }
        fprintf(stderr, "Could not populate event stock. Exiting...\n");
        PAPI_shutdown();
        return 0;
    }

    // Get the number of events contained in the event stock.
    nevts = num_evts(data);

    // Verify the validity of the cardinalities.
    cmbtotal = check_cards(params, &indexmemo, basenames, cards, ct, nevts, data);
    if(-1 == cmbtotal)
    {
        free(params.outputdir);
        remove_stock(data);
        if(READ_FROM_FILE == params.mode)
        {
            for(i = 0; i < ct; ++i)
            {
                free(basenames[i]);
            }
            free(basenames);
            free(cards);
        }
        PAPI_shutdown();
        return 0;
    }

    // Allocate enough space for all of the event+qualifier combinations.
    if (NULL == (allevts = (char**)malloc(cmbtotal*sizeof(char*)))) {
        fprintf(stderr, "Failed to allocate memory.\n");
        PAPI_shutdown();
        return 0;
    }

    // Create the qualifier combinations for each event.
    trav_evts(data, params.subsetsize, cards, nevts, ct, params.mode, allevts, &track, indexmemo, basenames);

    char *conf_file_name = ".cat_cfg";
    if( NULL != params.conf_file ) {
        conf_file_name = params.conf_file;
    }
    hw_desc_t *hw_desc = obtain_hardware_description(conf_file_name);

    /* Set the default number of threads to the OMP_NUM_THREADS environment
     * variable if it is defined. Otherwise, set it to the number of CPUs
     * in a single socket. */
    int numSetThreads = 1;
    char* envVarDefined = getenv("OMP_NUM_THREADS");
    if (NULL == envVarDefined) {
        omp_set_num_threads(hw_desc->numcpus);

        #pragma omp parallel default(shared)
        {
            if(!omp_get_thread_num()) {
                numSetThreads = omp_get_num_threads();
            }
        }

        if (numSetThreads != hw_desc->numcpus) {
            fprintf(stderr, "Warning! Failed to set default number of threads to number of CPUs in a single socket.\n");
        }
    }

    // Run the benchmark for each qualifier combination.
    testbench(allevts, cmbtotal, hw_desc, params, myid, nprocs);

    // Free dynamically allocated memory.
    free(params.outputdir);
    remove_stock(data);
    if(READ_FROM_FILE == params.mode)
    {
        for(i = 0; i < ct; ++i)
        {
            free(basenames[i]);
        }
        free(basenames);
        free(cards);
        free(indexmemo);
    }
    for(i = 0; i < cmbtotal; ++i)
    {
        free(allevts[i]);
    }
    free(allevts);
    free(hw_desc);

    PAPI_shutdown();

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;
}

unsigned long int omp_get_thread_num_wrapper() {
    return omp_get_thread_num();
}

// Verify that valid qualifier counts are provided and count their combinations.
int check_cards(cat_params_t params, int** indexmemo, char** basenames, int* cards, int ct, int nevts, evstock* data)
{
    int i, j, minim, n, cmbtotal = 0;
    char *name;
    int mode = params.mode;
    int pk = params.subsetsize;

    // User provided a file of events.
    if(READ_FROM_FILE == mode)
    {
        // Compute the total number of qualifier combinations and allocate memory to store them.
        if (NULL == ((*indexmemo) = (int*)malloc(ct*sizeof(int)))) {
            fprintf(stderr, "Failed to allocate memory.\n");
            return 0;
        }

        // Find the index in the main stock whose event corresponds to that in the file provided.
        // This simplifies looking up event qualifiers later.
        for(i = 0; i < ct; ++i)
        {
            if(NULL == basenames[i])
            {
                (*indexmemo)[i] = -1;
                cmbtotal -= 1;
                continue;
            }

            // j is the index of the event name provided by the user.
            for(j = 0; j < nevts; ++j)
            {
                name = evt_name(data, j);
                if(strcmp(basenames[i], name) == 0)
                {
                    break;
                }
            }

            // If the event name provided by the user does not match any of the main event
            // names in the architecture, then it either contains qualifiers or it does not
            // exist.
            if(cards[i] != 0 && j == nevts)
            {
                fprintf(stderr, "The provided event '%s' is either not in the architecture or contains qualifiers.\n" \
                        "If the latter, use '0' in place of the provided '%d'.\n", basenames[i], cards[i]);
                cards[i] = 0;
            }

            // If an invalid (negative) qualifier count was given, use zero qualifiers.
            if(cards[i] < 0)
            {
                fprintf(stderr, "The qualifier count (provided for event '%s') cannot be negative.\n", basenames[i]);
                cards[i] = 0;
            }

            (*indexmemo)[i] = j;
        }

        // Count the total number of events to test.
        for(i = 0; i < ct; ++i)
        {
            // If no qualifiers are used, then just count the event itself.
            if(cards[i] <= 0)
            {
                cmbtotal += 1;
                continue;
            }

            // Get the number of qualifiers which belong to the main event.
            if((*indexmemo)[i] != -1)
            {
                n = num_quals(data, (*indexmemo)[i]);
            }
            else
            {
                n = 0;
            }

            // If the user specifies to use more qualifiers than are available
            // for the main event, do not use any qualifiers. Otherwise, count
            // the number of combinations of qualifiers for the main event.
            minim = cards[i];
            if(cards[i] > n || cards[i] < 0)
            {
                minim = 0;
            }
            cmbtotal += comb(n, minim);
        }
    }
    // User wants to inspect all events in the architecture.
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            // Get the number of qualifiers which belong to the main event.
            n = num_quals(data, i);

            // If the user specifies to use more qualifiers than are available
            // for the main event, do not use any qualifiers. Otherwise, count
            // the number of combinations of qualifiers for the main event.
            minim = pk;
            if(pk > n || pk < 0)
            {
                minim = 0;
            }
            cmbtotal += comb(n, minim);
        }
    }

    return cmbtotal;
}

static hw_desc_t *obtain_hardware_description(char *conf_file_name){
    int i,j;
    hw_desc_t *hw_desc;
	PAPI_mh_level_t *L;
    const PAPI_hw_info_t *meminfo;

    // Allocate some space.
    hw_desc = (hw_desc_t *)calloc(1, sizeof(hw_desc_t));

    // Set at least the L1 cache size to a default value.
    hw_desc->dcache_line_size[0] = 64;

    // Set other default values.
    for( i=0; i<_MAX_SUPPORTED_CACHE_LEVELS; ++i ) {
        hw_desc->split[i] = 1;
        hw_desc->pts_per_reg[i] = 3;
    }
    hw_desc->mmsplit = 1;
    hw_desc->pts_per_mm = 3;
    hw_desc->maxPPB = 512;

    // Obtain hardware values through PAPI_get_hardware_info().
    meminfo = PAPI_get_hardware_info();
    if( NULL != meminfo ) {
        hw_desc->numcpus = meminfo->ncpu;
        hw_desc->cache_levels = meminfo->mem_hierarchy.levels;
	    L = ( PAPI_mh_level_t * ) & ( meminfo->mem_hierarchy.level[0] );
        for ( i = 0; i < meminfo->mem_hierarchy.levels && i<_MAX_SUPPORTED_CACHE_LEVELS; i++ ) {
            for ( j = 0; j < 2; j++ ) {
                if ( (PAPI_MH_TYPE_DATA == PAPI_MH_CACHE_TYPE(L[i].cache[j].type)) ||
                     (PAPI_MH_TYPE_UNIFIED == PAPI_MH_CACHE_TYPE(L[i].cache[j].type)) ){
                    hw_desc->dcache_line_size[i] = L[i].cache[j].line_size;
                    hw_desc->dcache_size[i] =      L[i].cache[j].size;
                    hw_desc->dcache_assoc[i] =     L[i].cache[j].associativity;
                }
                if ( (PAPI_MH_TYPE_INST == PAPI_MH_CACHE_TYPE(L[i].cache[j].type)) ||
                     (PAPI_MH_TYPE_UNIFIED == PAPI_MH_CACHE_TYPE(L[i].cache[j].type)) ){
                    hw_desc->icache_line_size[i] = L[i].cache[j].line_size;
                    hw_desc->icache_size[i] =      L[i].cache[j].size;
                    hw_desc->icache_assoc[i] =     L[i].cache[j].associativity;
                }
            }
        }
    }

    // Read the config file, if there, in case the user wants to overwrite some values.
    read_conf_file(conf_file_name, hw_desc);
    return hw_desc;
}



static int parse_line(FILE *input, char **key, long long *value){
    int status;
    size_t linelen=0, len;
    char *line=NULL;
    char *pos=NULL;

    // Read one line from the input file.
    int ret_val = (int)getline(&line, &linelen, input);
    if( ret_val < 0 )
        return ret_val;

    // Kill the part of the line after the comment character '#'.
    pos = strchr(line, '#');
    if( NULL != pos ){
        *pos = '\0';
    }

    // Make sure the line is an assignment.
    pos = strchr(line, '=');
    if( NULL == pos ){
        goto handle_error;
    }

    len = strcspn(line, " =");
    *key = (char *)calloc((1+len),sizeof(char));
    strncpy(*key, line, len);

    // Scan the line to make sure it has the form "key = value"
    status = sscanf(pos, "= %lld", value);
    if(1 != status){
        fprintf(stderr,"Malformed line in conf file: '%s'\n", line);
        goto handle_error;
    }

    return 0;

handle_error:
    free(line);
    key = NULL;
    *value = 0;
    line = NULL;
    linelen = 0;
    return 1;
}


static void read_conf_file(char *conf_file_name, hw_desc_t *hw_desc){
    FILE *input;

    // Try to open the file.
    input = fopen(conf_file_name, "r");
    if (NULL == input ){
        return;
    }

    while(1){
        long long value;
        char *key=NULL;

        int ret_val = parse_line(input, &key, &value);
        if( ret_val < 0 ){
            free(key);
            break;
        }else if( ret_val > 0 ){
            continue;
        }

        // If the user has set "AUTO_DISCOVERY_MODE = 1" then we don't need to process this file.
        // Otherwise, any entry in this file should overwrite what we auto discovered.
        if( !strcmp(key, "AUTO_DISCOVERY_MODE") && (value == 1) ){
            return;
        // Data caches (including unified caches)
        }else if( !strcmp(key, "L1_DCACHE_LINE_SIZE") || !strcmp(key, "L1_UCACHE_LINE_SIZE") ){
            hw_desc->dcache_line_size[0] = value;
        }else if( !strcmp(key, "L2_DCACHE_LINE_SIZE") || !strcmp(key, "L2_UCACHE_LINE_SIZE") ){
            hw_desc->dcache_line_size[1] = value;
        }else if( !strcmp(key, "L3_DCACHE_LINE_SIZE") || !strcmp(key, "L3_UCACHE_LINE_SIZE") ){
            hw_desc->dcache_line_size[2] = value;
        }else if( !strcmp(key, "L4_DCACHE_LINE_SIZE") || !strcmp(key, "L4_UCACHE_LINE_SIZE") ){
            hw_desc->dcache_line_size[3] = value;
        }else if( !strcmp(key, "L1_DCACHE_SIZE") || !strcmp(key, "L1_UCACHE_SIZE") ){
            if( hw_desc->cache_levels < 1 ) hw_desc->cache_levels = 1;
            hw_desc->dcache_size[0] = value;
        }else if( !strcmp(key, "L2_DCACHE_SIZE") || !strcmp(key, "L2_UCACHE_SIZE") ){
            if( hw_desc->cache_levels < 2 ) hw_desc->cache_levels = 2;
            hw_desc->dcache_size[1] = value;
        }else if( !strcmp(key, "L3_DCACHE_SIZE") || !strcmp(key, "L3_UCACHE_SIZE") ){
            if( hw_desc->cache_levels < 3 ) hw_desc->cache_levels = 3;
            hw_desc->dcache_size[2] = value;
        }else if( !strcmp(key, "L4_DCACHE_SIZE") || !strcmp(key, "L4_UCACHE_SIZE") ){
            if( hw_desc->cache_levels < 4 ) hw_desc->cache_levels = 4;
            hw_desc->dcache_size[3] = value;
        // Instruction caches (including unified caches)
        }else if( !strcmp(key, "L1_ICACHE_LINE_SIZE") || !strcmp(key, "L1_UCACHE_LINE_SIZE") ){
            hw_desc->icache_line_size[0] = value;
        }else if( !strcmp(key, "L2_ICACHE_LINE_SIZE") || !strcmp(key, "L2_UCACHE_LINE_SIZE") ){
            hw_desc->icache_line_size[1] = value;
        }else if( !strcmp(key, "L3_ICACHE_LINE_SIZE") || !strcmp(key, "L3_UCACHE_LINE_SIZE") ){
            hw_desc->icache_line_size[2] = value;
        }else if( !strcmp(key, "L4_ICACHE_LINE_SIZE") || !strcmp(key, "L4_UCACHE_LINE_SIZE") ){
            hw_desc->icache_line_size[3] = value;
        }else if( !strcmp(key, "L1_ICACHE_SIZE") || !strcmp(key, "L1_UCACHE_SIZE") ){
            hw_desc->icache_size[0] = value;
        }else if( !strcmp(key, "L2_ICACHE_SIZE") || !strcmp(key, "L2_UCACHE_SIZE") ){
            hw_desc->icache_size[1] = value;
        }else if( !strcmp(key, "L3_ICACHE_SIZE") || !strcmp(key, "L3_UCACHE_SIZE") ){
            hw_desc->icache_size[2] = value;
        }else if( !strcmp(key, "L4_ICACHE_SIZE") || !strcmp(key, "L4_UCACHE_SIZE") ){
            hw_desc->icache_size[3] = value;
        }else if( !strcmp(key, "L1_SPLIT") ){
            hw_desc->split[0] = value;
        }else if( !strcmp(key, "L2_SPLIT") ){
            hw_desc->split[1] = value;
        }else if( !strcmp(key, "L3_SPLIT") ){
            hw_desc->split[2] = value;
        }else if( !strcmp(key, "L4_SPLIT") ){
            hw_desc->split[3] = value;
        }else if( !strcmp(key, "MM_SPLIT") ){
            hw_desc->mmsplit = value;
        }else if( !strcmp(key, "PTS_PER_L1") ){
            hw_desc->pts_per_reg[0] = value;
        }else if( !strcmp(key, "PTS_PER_L2") ){
            hw_desc->pts_per_reg[1] = value;
        }else if( !strcmp(key, "PTS_PER_L3") ){
            hw_desc->pts_per_reg[2] = value;
        }else if( !strcmp(key, "PTS_PER_L4") ){
            hw_desc->pts_per_reg[3] = value;
        }else if( !strcmp(key, "PTS_PER_MM") ){
            hw_desc->pts_per_mm = value;
        }else if( !strcmp(key, "MAX_PPB") ){
            hw_desc->maxPPB = value;
        }

        free(key);
        key = NULL;
    }
    fclose(input);
    return;
}

// Read the contents of the file supplied by the user.
int setup_evts(char* inputfile, char*** basenames, int** evnt_cards)
{
    size_t linelen = 0;
    int cnt = 0, status = 0;
    char *line = NULL, *place;
    FILE *input;
    int evnt_count = 256;
  
    char **names = (char **)calloc(evnt_count, sizeof(char *));
    int *cards = (int *)calloc(evnt_count, sizeof(int));

    if (NULL == names || NULL == cards) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return 0;
    }

    // Read the base event name and cardinality columns.
    input = fopen(inputfile, "r");
    for(cnt=0; 1; cnt++)
    {
        ssize_t ret_val = getline(&line, &linelen, input);
        if( ret_val < 0 )
            break;
        if( cnt >= evnt_count )
        {
            evnt_count *= 2;
            names = realloc(names, evnt_count*sizeof(char *));
            cards = realloc(cards, evnt_count*sizeof(int));

            if (NULL == names || NULL == cards) {
                fprintf(stderr, "Failed to allocate memory.\n");
                return 0;
            }
        }

        place = strstr(line, " ");

        // If this line was commented, silently ignore it.
        if(strlen(line) > 0 && line[0] == '#') {
            names[cnt] = NULL;
            cards[cnt] = -1;
            cnt--;

            free(line);
            line = NULL;
            linelen = 0;
            continue;
        } else if( NULL == place ) {
            fprintf(stderr,"problem with line: '%s'\n",line);
            names[cnt] = NULL;
            cards[cnt] = -1;
            cnt--;

            free(line);
            line = NULL;
            linelen = 0;
            continue;
        }

        names[cnt] = NULL;
        status = sscanf(line, "%ms %d", &(names[cnt]), &(cards[cnt]) );

        // If this line was malformed, ignore it.
        if(2 != status)
        {
            fprintf(stderr,"problem with line: '%s'\n",line);
            names[cnt] = NULL;
            cards[cnt] = -1;
            cnt--;
        }

        free(line);
        line = NULL;
        linelen = 0;
    }
    free(line);
    fclose(input);

    *basenames = names;
    *evnt_cards = cards;

    return cnt;
}

// Recursively builds the list of all combinations of an event's qualifiers.
void combine_qualifiers(int n, int pk, int ct, char** list, char* name, char** allevts, int* track, int flag, int* bitmap)
{
    int original;
    int counter;
    int i;

    // Set flag in the array.
    original = bitmap[ct];
    bitmap[ct] = flag;

    // Only make recursive calls if there are more items.
    // Ensure proper cardinality.
    counter = 0;
    for(i = 0; i < n; ++i)
    {
        counter += bitmap[i];
    }    

    // Cannot use more qualifiers than are available.
    if(ct+1 < n)
    {
        // Make recursive calls both with and without a given qualifier.
        // Recursion cannot exceed the number of qualifiers specified by
        // the user.
        if(counter < pk)
        {
            combine_qualifiers(n, pk, ct+1, list, name, allevts, track, 1, bitmap); 
        }
        combine_qualifiers(n, pk, ct+1, list, name, allevts, track, 0, bitmap);
    }
    // Qualifier count matches that specified by the user.
    else
    {
        if(counter == pk)
        {
            // Construct the qualifier combination string.
            char* chunk;
            size_t evtsize = strlen(name)+1;
            for(i = 0; i < n; ++i)
            {
                if(bitmap[i] == 1)
                {
                    // Add one to account for the colon in front of the qualifier.
                    evtsize += strlen(list[i])+1;
                }
            }

            if (NULL == (chunk = (char*)malloc((evtsize+1)*sizeof(char)))) {
                fprintf(stderr, "Failed to allocate memory.\n");
                return;
            }

            strcpy(chunk,name);
            for(i = 0; i < n; ++i)
            {
                if(bitmap[i] == 1)
                {
                    strcat(chunk,":");
                    strcat(chunk,list[i]);
                }
            }

            // Add qualifier combination string to the list.
            allevts[*track] = strdup(chunk);
            *track += 1;

            free(chunk);
        }
    }

    // Undo effect of recursive call to combine other qualifiers.
    bitmap[ct] = original;

    return;
}

// Create the combinations of qualifiers for the events.
void trav_evts(evstock* stock, int pk, int* cards, int nevts, int selexnsize, int mode, char** allevts, int* track, int* indexmemo, char** basenames)
{
    int i, j, k, n = 0;
    char** chosen = NULL;
    char* name = NULL;
    int* bitmap = NULL;

    // User provided a file of events.
    if(READ_FROM_FILE == mode)
    {
        for(i = 0; i < selexnsize; ++i)
        {
            // Iterate through whole stock. If there are matches, proceed normally using the given cardinalities.
            j = indexmemo[i];
            if( -1 == j )
            {
                allevts[i] = NULL;
                continue;
            }

            // Get event's name and qualifier count.
            if(j == nevts)
            {
                // User a provided specific qualifier combination.
                name = basenames[i];
            }
            else
            {
                name = evt_name(stock, j);
                n = num_quals(stock, j);
            }

            // Create a list to contain the qualifiers.
            if(cards[i] > 0)
            {
                chosen = (char**)malloc(n*sizeof(char*));
                bitmap = (int*)calloc(n, sizeof(int));    
                
                if (NULL == chosen || NULL == bitmap) {
                    fprintf(stderr, "Failed to allocate memory.\n");
                    return;
                }

                // Store the qualifiers for the current event.
                for(k = 0; k < n; ++k)
                {
                    chosen[k] = strdup(stock->evts[j][k]);
                }
            }

            // Get combinations of all current event's qualifiers.
            if (n!=0 && cards[i]>0)
            {
                combine_qualifiers(n, cards[i], 0, chosen, name, allevts, track, 0, bitmap);
                combine_qualifiers(n, cards[i], 0, chosen, name, allevts, track, 1, bitmap);
            }
            else
            {
                allevts[*track] = strdup(name);
                *track += 1;
            }

            // Free the space back up.
            if(cards[i] > 0)
            {
                for(k = 0; k < n; ++k)
                {
                    free(chosen[k]);
                }
                free(chosen);
                free(bitmap);
            }
        }
    }
    // User wants to inspect all events in the architecture.
    else
    {
        for(i = 0; i < nevts; ++i)
        {
            // Get event's name and qualifier count.
            n = num_quals(stock, i);
            name = evt_name(stock, i);

            // Show progress to the user.
            //fprintf(stderr, "CURRENT EVENT: %s (%d/%d)\n", name, (i+1), nevts);

            // Create a list to contain the qualifiers.
            chosen = (char**)malloc(n*sizeof(char*));
            bitmap = (int*)calloc(n, sizeof(int));    

            if (NULL == chosen || NULL == bitmap) {
                fprintf(stderr, "Failed to allocate memory.\n");
                return;
            }

            // Store the qualifiers for the current event.
            for(j = 0; j < n; ++j)
            {
                chosen[j] = strdup(stock->evts[i][j]);
            }

            // Get combinations of all current event's qualifiers.
            if (n!=0)
            {
                combine_qualifiers(n, pk, 0, chosen, name, allevts, track, 0, bitmap);
                combine_qualifiers(n, pk, 0, chosen, name, allevts, track, 1, bitmap);
            }
            else
            {
                allevts[*track] = strdup(name);
                *track += 1;
            }

            // Free the space back up.
            for(j = 0; j < n; ++j)
            {
                free(chosen[j]);
            }
            free(chosen);
            free(bitmap);
        }
    }

    return;
}

// Compute the permutations of k objects from a set of n objects.
int perm(int n, int k)
{
    int i;
    int prod = 1;
    int diff = n-k;

    for(i = n; i > diff; --i)
    {
        prod *= i;
    }

    return prod;
}

// Compute the combinations of k objects from a set of n objects.
int comb(int n, int k)
{
    return perm(n, k)/perm(k, k);
}

static void print_progress(int prg)
{
    if(prg < 100)
        printf("%3d%%\b\b\b\b",prg);
    else
        printf("%3d%%\n",prg);

    fflush(stdout);
}

static void print_progress2(int prg)
{
    if(prg < 100)
        printf("Total:%3d%%  Current test:  0%%\b\b\b\b",prg);
    else
        printf("Total:%3d%%\n",prg);

    fflush(stdout);
}

void testbench(char** allevts, int cmbtotal, hw_desc_t *hw_desc, cat_params_t params, int myid, int nprocs)
{
    int i;
    int junk=((int)getpid()+123)/456;
    int low = myid*(cmbtotal/nprocs);
    int cap = (myid+1)*(cmbtotal/nprocs);
    int offset = nprocs*(1+cmbtotal/nprocs)-cmbtotal;

    // Divide the work as evenly as possible.
    if(myid >= offset) {
        cap += myid-offset+1;
        low += myid-offset;
    }

    // Make sure the user provided events and iterate through all events.
    if( 0 == cmbtotal )
    {
        fprintf(stderr, "No events to measure.\n");
        return;
    }

    // Run the branch benchmark by default if none are specified.
    if( 0 == params.bench_type )
    {
        params.bench_type |= BENCH_BRANCH;
        fprintf(stderr, "Warning: No benchmark specified. Running 'branch' by default.\n");
    }

    /* Benchmark I - Branch*/
    if( params.bench_type & BENCH_BRANCH )
    {
        if(params.show_progress) printf("Branch Benchmarks: ");

        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                branch_driver(allevts[i], junk, hw_desc, params.outputdir);
        }
        if(params.show_progress) print_progress(100);
    }

    /* Benchmark II - Data Cache Reads*/
    if( params.bench_type & BENCH_DCACHE_READ )
    {
        if ( !params.quick && 0 == myid )
        {
            if(params.show_progress)
            {
                printf("D-Cache Latencies:  0%%\b\b\b\b");
                fflush(stdout);
            }
            d_cache_driver("cat::latencies", params, hw_desc, 1, 0);
            if(params.show_progress) printf("\n");
        }

        if(params.show_progress) printf("D-Cache Read Benchmarks: ");
        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL ) {
                d_cache_driver(allevts[i], params, hw_desc, 0, 0);
            }
        }
        if(params.show_progress) print_progress2(100);
    }

    /* Benchmark III - Data Cache Writes*/
    if( params.bench_type & BENCH_DCACHE_WRITE )
    {
        // If the READ benchmark was run, do not recompute the latencies.
        if ( !(params.bench_type & BENCH_DCACHE_READ) && !params.quick)
        {
            if(params.show_progress)
            {
                printf("D-Cache Latencies:  0%%\b\b\b\b");
                fflush(stdout);
            }
            d_cache_driver("cat::latencies", params, hw_desc, 1, 0);
            if(params.show_progress) printf("\n");
        }

        if(params.show_progress) printf("D-Cache Write Benchmarks: ");
        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL ) {
                d_cache_driver(allevts[i], params, hw_desc, 0, 1);
            }
        }
        if(params.show_progress) print_progress2(100);
    }

    /* Benchmark IV - FLOPS*/
    if( params.bench_type & BENCH_FLOPS )
    {
        if(params.show_progress) printf("FLOP Benchmarks: ");

        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                flops_driver(allevts[i], hw_desc, params.outputdir);
        }
        if(params.show_progress) print_progress(100);
    }

    /* Benchmark V - Instruction Cache*/
    if( params.bench_type & BENCH_ICACHE_READ )
    {
        if(params.show_progress) printf("I-Cache Benchmarks: ");

        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress2((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                i_cache_driver(allevts[i], junk, hw_desc, params.outputdir, params.show_progress);
        }
        if(params.show_progress) print_progress2(100);
    }

    /* Benchmark VI - Vector FLOPS*/
    if( params.bench_type & BENCH_VEC )
    {
        if(params.show_progress) printf("Vector FLOP Benchmarks: ");

        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                vec_driver(allevts[i], hw_desc, params.outputdir);
        }
        if(params.show_progress) print_progress(100);
    }

    /* Benchmark VII - Instructions*/
    if( params.bench_type & BENCH_INSTR )
    {
        if(params.show_progress) printf("Instruction Benchmarks: ");

        for(i = low; i < cap; ++i)
        {
            if(params.show_progress) print_progress((100*i)/cmbtotal);

            if( allevts[i] != NULL )
                instr_driver(allevts[i], hw_desc, params.outputdir);
        }
        if(params.show_progress) print_progress(100);
    }

    return;
}

int parseArgs(int argc, char **argv, cat_params_t *params){
    char *name = argv[0];
    char *tmp = NULL;
    int dirlen = 0;
    int kflag  = 0;
    int inflag = 0;
    FILE *test = NULL;
    int len, status = 0;

    params->subsetsize = -1;

    // Parse the command line arguments
    while(--argc){
        ++argv;
        if( !strcmp(argv[0],"-h") ){
            print_usage(name);
            return -1;
        }
        if( argc > 1 && !strcmp(argv[0],"-k") ){
            params->subsetsize = atoi(argv[1]);
            if( params->subsetsize < 0 )
            {
                params->subsetsize = 0;
                fprintf(stderr, "Warning: Cannot pass a negative value to -k.\n");
            }
            params->mode = USE_ALL_EVENTS;
            kflag = 1;
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-n") ){
            params->max_iter = atoi(argv[1]);
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-conf") ){
            params->conf_file = argv[1];
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-in") ){
            params->inputfile = argv[1];
            params->mode = READ_FROM_FILE;
            inflag = 1;
            --argc;
            ++argv;
            continue;
        }
        if( argc > 1 && !strcmp(argv[0],"-out") ){
            tmp = argv[1];
            --argc;
            ++argv;
            continue;
        }
        if( !strcmp(argv[0],"-verbose") ){
            params->show_progress = 1;
            continue;
        }
        if( !strcmp(argv[0],"-quick") ){
            params->quick = 1;
            continue;
        }
        if( !strcmp(argv[0],"-branch") ){
            params->bench_type |= BENCH_BRANCH;
            continue;
        }
        if( !strcmp(argv[0],"-dcr") ){
            params->bench_type |= BENCH_DCACHE_READ;
            continue;
        }
        if( !strcmp(argv[0],"-dcw") ){
            params->bench_type |= BENCH_DCACHE_WRITE;
            continue;
        }
        if( !strcmp(argv[0],"-flops") ){
            params->bench_type |= BENCH_FLOPS;
            continue;
        }
        if( !strcmp(argv[0],"-ic") ){
            params->bench_type |= BENCH_ICACHE_READ;
            continue;
        }
        if( !strcmp(argv[0],"-vec") ){
            params->bench_type |= BENCH_VEC;
            continue;
        }
        if( !strcmp(argv[0],"-instr") ){
            params->bench_type |= BENCH_INSTR;
            continue;
        }

        print_usage(name);
        return -1;
    }

    // MODE INFO: mode 1 uses file; mode 2 uses all native events.
    if(READ_FROM_FILE == params->mode)
    {
        test = fopen(params->inputfile, "r");
        if(test == NULL)
        {
            fprintf(stderr, "Could not open %s. Exiting...\n", params->inputfile);
            return -1;
        }
        fclose(test);
    }

    // Make sure user does not specify both modes simultaneously.
    if(kflag == 1 && inflag == 1)
    {
        fprintf(stderr, "Cannot use -k flag with -in flag. Exiting...\n");
        return -1;
    }

    // Make sure user specifies mode explicitly.
    if(kflag == 0 && inflag == 0)
    {
        print_usage(name);
        return -1;
    }

    // Make sure output path was provided.
    if(tmp == NULL)
    {
        fprintf(stderr, "Output path not provided. Exiting...\n");
        return -1;
    }

    // Write output files in the user-specified directory.
    dirlen = strlen(tmp);
    params->outputdir = (char*)malloc((2+dirlen)*sizeof(char));

    if (NULL == params->outputdir) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return -1;
    }

    len = snprintf( params->outputdir, 2+dirlen, "%s/", tmp);
    if( len < 1+dirlen )
    {
        fprintf(stderr, "Problem with output directory name.\n");
        return -1;
    }

    // Make sure files can be written to the provided path.
    status = access(params->outputdir, W_OK);
    if(status != 0)
    {
        fprintf(stderr, "Permission to write files to \"%s\" denied. Make sure the path exists and is writable.\n", tmp);
        return -1;
    }

    return 0;
}

// Show the user how to properly use the program.
void print_usage(char* name)
{
    fprintf(stdout, "\nUsage: %s [OPTIONS...]\n", name);

    fprintf(stdout, "\nRequired:\n");
    fprintf(stdout, "  -out <path>   Output files location.\n");
    fprintf(stdout, "  -in  <file>   Events and cardinalities file.\n");
    fprintf(stdout, "  -k   <value>  Cardinality of subsets.\n");
    fprintf(stdout, "  Parameters \"-k\" and \"-in\" are mutually exclusive.\n");
    
    fprintf(stdout, "\nOptional:\n");
    fprintf(stdout, "  -conf    <path>   Configuration file location.\n");
    fprintf(stdout, "  -verbose          Show benchmark progress in the standard output.\n");
    fprintf(stdout, "  -quick            Skip latency tests.\n");
    fprintf(stdout, "  -n       <value>  Number of iterations for data cache kernels.\n");
    fprintf(stdout, "  -branch           Branch kernels.\n");
    fprintf(stdout, "  -dcr              Data cache reading kernels.\n");
    fprintf(stdout, "  -dcw              Data cache writing kernels.\n");
    fprintf(stdout, "  -flops            Floating point operations kernels.\n");
    fprintf(stdout, "  -ic               Instruction cache kernels.\n");
    fprintf(stdout, "  -vec              Vector FLOPs kernels.\n");
    fprintf(stdout, "  -instr            Instructions kernels.\n");

    fprintf(stdout, "\n");
    fprintf(stdout, "EXAMPLE: %s -in event_list.txt -out OUTPUT_DIRECTORY -branch -dcw\n", name);
    fprintf(stdout, "\n");

    return;
}
