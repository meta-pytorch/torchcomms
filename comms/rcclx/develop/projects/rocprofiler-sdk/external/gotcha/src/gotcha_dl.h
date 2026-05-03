#ifndef GOTCHA_DL_H
#define GOTCHA_DL_H

#include "gotcha/gotcha_types.h"
#include "hash.h"
#include "tool.h"

void handle_libdl(const gotcha_init_config_t *config);
extern void update_all_library_gots(hash_table_t *bindings);
extern long lookup_exported_symbol(const char *name, const struct link_map *lib,
                                   void **symbol);
extern int prepare_symbol(struct internal_binding_t *binding);
extern void **getInternalBindingAddressPointer(struct internal_binding_t **in);

extern gotcha_wrappee_handle_t orig_dlopen_handle;
extern gotcha_wrappee_handle_t orig_dlsym_handle;

extern struct gotcha_binding_t dl_binds[];
#endif
