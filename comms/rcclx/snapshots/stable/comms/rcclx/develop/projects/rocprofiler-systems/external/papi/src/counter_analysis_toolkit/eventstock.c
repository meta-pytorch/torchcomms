#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "papi.h"
#include "eventstock.h"

#if !defined(_PAPI_CPU_COMPONENT_NAME)
#define _PAPI_CPU_COMPONENT_NAME "perf_event"
#endif

int build_stock(evstock* stock)
{
    int ret;
    PAPI_event_info_t info;
    int cid;
    int ncomps = PAPI_num_components();
    int event_counter = 0;
    int subctr = 0;
    int tmp_event_count;
    int event_qual_i, event_i;

    if (!stock) return 1;

    event_i = 0 | PAPI_NATIVE_MASK;

    // Add the names to the stock.
    event_counter = 0;
    for(cid = 0; cid < ncomps; ++cid)
    {
        const PAPI_component_info_t *cmp_info = PAPI_get_component_info(cid);
        if( strcmp(cmp_info->name, _PAPI_CPU_COMPONENT_NAME) )
            continue;

        if (cmp_info->disabled == PAPI_EDELAY_INIT) {
            int nvt_code = 0 | PAPI_NATIVE_MASK;
            PAPI_enum_cmp_event(&nvt_code, PAPI_ENUM_FIRST, cid);
        }

        tmp_event_count = cmp_info->num_native_events;

        // Set the data stock's sizes all to zero.
        if (NULL == (stock->evtsizes = (int*)calloc( (tmp_event_count),sizeof(int) ))) {
            fprintf(stderr, "Failed allocation of stock->evtsizes.\n");
            goto gracious_error;
        }

        if (NULL == (stock->base_evts = (char**)malloc( (tmp_event_count)*sizeof(char*) ))) {
            fprintf(stderr, "Failed allocation of stock->base_evts.\n");
            goto gracious_error;
        }

        if (NULL == (stock->evts = (char***)malloc((tmp_event_count)*sizeof(char**)))) {
            fprintf(stderr, "Failed allocation of stock->evts.\n");
            goto gracious_error;
        }

        if (NULL == (stock->maxqualsize = (size_t *)calloc( tmp_event_count, sizeof(size_t) ))) {
            fprintf(stderr, "Failed allocation of stock->maxqualsize.\n");
            goto gracious_error;
        }

        break;
    }

    if( 0 == tmp_event_count ){
        fprintf(stderr,"ERROR: CPU component (%s) not found. Exiting.\n",_PAPI_CPU_COMPONENT_NAME);
        goto gracious_error;
    }

    // At this point "cid" contains the id of the perf_event (CPU) component.

    ret=PAPI_enum_cmp_event(&event_i,PAPI_ENUM_FIRST,cid);
    if(ret!=PAPI_OK){
        return 0;
    }

    do{
        int i, max_qual_count = 32;
        size_t max_qual_len, tmp_qual_len;
        memset(&info,0,sizeof(info));
        event_qual_i = event_i;

        // Resize the arrays if needed.
        if( event_counter >= tmp_event_count ){
            tmp_event_count *= 2;
            stock->evts = (char ***)realloc( stock->evts, tmp_event_count*sizeof(char **) );
            stock->evtsizes = (int *)realloc( stock->evtsizes, tmp_event_count*sizeof(int) );
            stock->base_evts = (char **)realloc( stock->base_evts, tmp_event_count*sizeof(char *) );
            stock->maxqualsize = (size_t *)realloc( stock->maxqualsize, tmp_event_count*sizeof(size_t) );
        }

        if (NULL == (stock->evts[event_counter] = (char**)malloc( max_qual_count*sizeof(char*) )) ) {
            fprintf(stderr, "Failed allocation of stock->evts[i].\n");
            goto gracious_error;
        }

        max_qual_len = 0;
        subctr = 0;
        i = 0;

        do
        {
            char *col_pos;
            ret=PAPI_get_event_info(event_qual_i,&info);
            if(ret != PAPI_OK)
                continue;

            if( 0 == i ){
                // The first iteration of the inner do loop will give us
                // the base event, without qualifiers.
                stock->base_evts[event_counter] = strdup(info.symbol);
                i++;
                continue;
            }

            // TODO: For the CPU component, we skip qualifiers that
            // contain the string "=". This assumption should be
            // removed when working with other components.
            if( NULL != strstr(info.symbol, "=") )
                continue;

            col_pos = rindex(info.symbol, ':');
            if ( NULL == col_pos ){
                continue;
            }

            // Resize the array of qualifiers as needed.
            if( subctr >= max_qual_count ){
                max_qual_count *= 2;
                stock->evts[event_counter] = (char **)realloc( stock->evts[event_counter], max_qual_count*sizeof(char *) );
            }

            // Copy the qualifier name into the array.
            stock->evts[event_counter][subctr] = strdup(col_pos+1);
            tmp_qual_len = strlen( stock->evts[event_counter][subctr] ) + 1;
            if( tmp_qual_len > max_qual_len )
                max_qual_len = tmp_qual_len;
            subctr++;

        } while(PAPI_enum_cmp_event(&event_qual_i,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK);
        stock->evtsizes[event_counter] = subctr;
        stock->maxqualsize[event_counter] = max_qual_len;
        event_counter++;
    } while( PAPI_enum_cmp_event(&event_i,PAPI_ENUM_EVENTS,cid)==PAPI_OK );

    stock->size = event_counter;
    return 0;

gracious_error:
    // Frees only the successfully allocated arrays
    remove_stock(stock);
    return 1;
}

void print_stock(evstock* stock)
{
    int i, j;
    for(i = 0; i < stock->size; ++i)
    {
        fprintf(stdout, "BASE EVENT <%s>\n", stock->base_evts[i]);
        for(j = 0; j < stock->evtsizes[i]; ++j)
        {
            fprintf(stdout, "%s\n", stock->evts[i][j]);
        }
    }

    return;
}

int num_evts(evstock* stock)
{
    return stock->size;
}

int num_quals(evstock* stock, int base_evt)
{
    return stock->evtsizes[base_evt];
}

size_t max_qual_size(evstock* stock, int base_evt)
{
    return stock->maxqualsize[base_evt];
}

char* evt_qual(evstock* stock, int base_evt, int tag)
{
    return stock->evts[base_evt][tag];
}

char* evt_name(evstock* stock, int index)
{
    return stock->base_evts[index];
}

void remove_stock(evstock* stock)
{
    if (!stock) return;

    int i, j;
    for(i = 0; i < stock->size; ++i)
    {
        if (!stock->evtsizes)
        for(j = 0; j < stock->evtsizes[i]; ++j)
        {
            if (stock->evts[i][j])
                free(stock->evts[i][j]);
        }
        if (stock->evts[i])
            free(stock->evts[i]);
        if (stock->base_evts[i])
            free(stock->base_evts[i]);
    }
    if (stock->evts)
        free(stock->evts);
    if (stock->base_evts)
        free(stock->base_evts);
    if (stock->evtsizes)
        free(stock->evtsizes);
    if (stock->maxqualsize)
        free(stock->maxqualsize);
    free(stock);

    return;
}
