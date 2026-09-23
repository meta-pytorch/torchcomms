#ifndef _EVENT_STOCK_
#define _EVENT_STOCK_

typedef struct
{
    int size;
    int* evtsizes;
    size_t* maxqualsize;
    char** base_evts;
    char*** evts;
} evstock;

int     build_stock(evstock* stock);
void    print_stock(evstock* stock);
int     num_evts(evstock* stock);
int     num_quals(evstock* stock, int base_evt);
size_t  max_qual_size(evstock* stock, int base_evt);
char*   evt_qual(evstock* stock, int base_evt, int tag);
char*   evt_name(evstock* stock, int index);
void    remove_stock(evstock* stock);

#endif
