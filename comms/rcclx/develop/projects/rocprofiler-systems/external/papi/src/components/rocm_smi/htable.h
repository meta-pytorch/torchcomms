/**
 * @file    htable.h
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 */
#ifndef __HTABLE_H__
#define __HTABLE_H__
#include <string.h>
#include <inttypes.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#define HTABLE_NEEDS_TO_GROW(table)   (table->size > 0 && table->capacity / table->size < 2)
#define HTABLE_NEEDS_TO_SHRINK(table) (table->size > 0 && table->capacity / table->size > 8)

#define HTABLE_SUCCESS ( 0)
#define HTABLE_ENOVAL  (-1)
#define HTABLE_EINVAL  (-2)
#define HTABLE_ENOMEM  (-3)

struct hash_table_entry {
    char *key;
    void *val;
    struct hash_table_entry *next;
};

struct hash_table {
    uint32_t capacity;
    uint32_t size;
    struct hash_table_entry **buckets;
};

static uint64_t hash_func(const char *);

static int create_table(uint64_t, struct hash_table **);
static int destroy_table(struct hash_table *);
static int rehash_table(struct hash_table *, struct hash_table *);
static int move_table(struct hash_table *, struct hash_table *);
static int check_n_resize_table(struct hash_table *);
static int destroy_table_entries(struct hash_table *);

static int create_table_entry(const char *, void *,
                              struct hash_table_entry **);
static int destroy_table_entry(struct hash_table_entry *);
static int insert_table_entry(struct hash_table *, struct hash_table_entry *);
static int delete_table_entry(struct hash_table *, struct hash_table_entry *);
static int find_table_entry(struct hash_table *, const char *,
                            struct hash_table_entry **);

static inline int
htable_init(void **handle)
{
    int htable_errno = HTABLE_SUCCESS;

#define HTABLE_MIN_SIZE (8)
    struct hash_table *table = NULL;
    htable_errno = create_table(HTABLE_MIN_SIZE, &table);
    if (htable_errno != HTABLE_SUCCESS) {
        goto fn_fail;
    }

    *handle = table;

  fn_exit:
    return htable_errno;
  fn_fail:
    *handle = NULL;
    goto fn_exit;
}

static inline int
htable_shutdown(void *handle)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *table = (struct hash_table *) handle;

    if (table == NULL) {
        return HTABLE_EINVAL;
    }

    destroy_table_entries(table);
    destroy_table(table);

    return htable_errno;
}

static inline int
htable_insert(void *handle, const char *key, void *in)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *table = (struct hash_table *) handle;

    if (table == NULL || key == NULL) {
        return HTABLE_EINVAL;
    }

    struct hash_table_entry *entry = NULL;
    htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno == HTABLE_SUCCESS) {
        entry->val = in;
        goto fn_exit;
    }

    htable_errno = create_table_entry(key, in, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        goto fn_fail;
    }

    htable_errno = insert_table_entry(table, entry);
    if (htable_errno != HTABLE_SUCCESS) {
        goto fn_fail;
    }

    htable_errno = check_n_resize_table(table);

  fn_exit:
    return htable_errno;
  fn_fail:
    if (entry) {
        papi_free(entry);
    }
    goto fn_exit;
}

static inline int
htable_delete(void *handle, const char *key)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *table = (struct hash_table *) handle;

    if (table == NULL || key == NULL) {
        return HTABLE_EINVAL;
    }

    struct hash_table_entry *entry = NULL;
    htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }

    entry->val = NULL;

    htable_errno = delete_table_entry(table, entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }

    htable_errno = destroy_table_entry(entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }

    return check_n_resize_table(table);
}

static inline int
htable_find(void *handle, const char *key, void **out)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *table = (struct hash_table *) handle;

    if (table == NULL || key == NULL || out == NULL) {
        return HTABLE_EINVAL;
    }

    struct hash_table_entry *entry = NULL;
    htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }

    *out = entry->val;
    return htable_errno;
}

/**
 * djb2 hash function
 */
uint64_t
hash_func(const char *string)
{
    uint64_t hash = 5381;
    int c;
    while ((c = *string++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

int
create_table(uint64_t size, struct hash_table **table)
{
    int htable_errno = HTABLE_SUCCESS;

    *table = papi_calloc(1, sizeof(**table));
    if (table == NULL) {
        htable_errno = HTABLE_ENOMEM;
        goto fn_exit;
    }

    (*table)->buckets = papi_calloc(size, sizeof(*(*table)->buckets));
    if ((*table)->buckets == NULL) {
        htable_errno = HTABLE_ENOMEM;
        goto fn_exit;
    }

    (*table)->capacity = size;

  fn_exit:
    return htable_errno;
}

int
destroy_table(struct hash_table *table)
{
    int htable_errno = HTABLE_SUCCESS;

    if (table && table->buckets) {
        papi_free(table->buckets);
    }

    if (table) {
        papi_free(table);
    }

    return htable_errno;
}

int
rehash_table(struct hash_table *old_table, struct hash_table *new_table)
{
    uint64_t old_id;
    for (old_id = 0; old_id < old_table->capacity; ++old_id) {
        struct hash_table_entry *entry = old_table->buckets[old_id];
        struct hash_table_entry *next;
        while (entry) {
            next = entry->next;
            delete_table_entry(old_table, entry);
            insert_table_entry(new_table, entry);
            entry = next;
        }
    }

    return HTABLE_SUCCESS;
}

int
move_table(struct hash_table *new_table, struct hash_table *old_table)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table_entry **old_buckets = old_table->buckets;

    old_table->capacity = new_table->capacity;
    old_table->size = new_table->size;
    old_table->buckets = new_table->buckets;
    new_table->buckets = NULL;
    papi_free(old_buckets);

    return htable_errno;
}

int
destroy_table_entries(struct hash_table *table)
{
    int htable_errno = HTABLE_SUCCESS;
    uint64_t i;

    for (i = 0; i < table->capacity; ++i) {
        struct hash_table_entry *entry = table->buckets[i];
        struct hash_table_entry *tmp = NULL;

        while (entry) {
            tmp = entry;
            entry = entry->next;
            delete_table_entry(table, tmp);
            destroy_table_entry(tmp);
        }
    }

    return htable_errno;
}

int
check_n_resize_table(struct hash_table *table)
{
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *new_table = NULL;
    char resize =
        (HTABLE_NEEDS_TO_GROW(table) << 1) | HTABLE_NEEDS_TO_SHRINK(table);

    if (resize) {
        uint64_t new_capacity = (resize & 0x2) ?
            table->capacity * 2 : table->capacity / 2;
        htable_errno = create_table(new_capacity, &new_table);
        if (htable_errno != HTABLE_SUCCESS) {
            goto fn_fail;
        }

        htable_errno = rehash_table(table, new_table);
        if (htable_errno != HTABLE_SUCCESS) {
            goto fn_fail;
        }

        move_table(new_table, table);
        destroy_table(new_table);
    }

  fn_exit:
    return htable_errno;
  fn_fail:
    if (new_table) {
        destroy_table(new_table);
    }
    goto fn_exit;
}

int
create_table_entry(const char *key, void *val, struct hash_table_entry **entry)
{
    int htable_errno = HTABLE_SUCCESS;

    *entry = papi_calloc(1, sizeof(**entry));
    if (*entry == NULL) {
        return HTABLE_ENOMEM;
    }
    (*entry)->key = strdup(key);
    (*entry)->val = val;
    (*entry)->next = NULL;

    return htable_errno;
}

int
destroy_table_entry(struct hash_table_entry *entry)
{
    int htable_errno = HTABLE_SUCCESS;
    papi_free(entry->key);
    papi_free(entry);
    return htable_errno;
}

int
insert_table_entry(struct hash_table *table, struct hash_table_entry *entry)
{
    int htable_errno = HTABLE_SUCCESS;

    uint64_t id = hash_func(entry->key) % table->capacity;

    if (table->buckets[id]) {
        entry->next = table->buckets[id];
    }

    table->buckets[id] = entry;
    ++table->size;

    return htable_errno;
}

int
delete_table_entry(struct hash_table *table, struct hash_table_entry *entry)
{
    int htable_errno = HTABLE_SUCCESS;

    uint64_t id = hash_func(entry->key) % table->capacity;

    if (table->buckets[id] == entry) {
        table->buckets[id] = entry->next;
        entry->next = NULL;
        goto fn_exit;
    }

    struct hash_table_entry *prev = table->buckets[id];
    struct hash_table_entry *curr = table->buckets[id]->next;

    while (curr) {
        if (curr == entry) {
            prev->next = curr->next;
            curr->next = NULL;
            break;
        }
        prev = prev->next;
        curr = curr->next;
    }

  fn_exit:
    --table->size;
    return htable_errno;
}

int
find_table_entry(struct hash_table *table, const char *key,
                 struct hash_table_entry **entry)
{
    int htable_errno;

    uint64_t id = hash_func(key) % table->capacity;
    struct hash_table_entry *head = table->buckets[id];
    if (head == NULL) {
        htable_errno = HTABLE_ENOVAL;
        goto fn_exit;
    }

    struct hash_table_entry *curr = head;
    while (curr && strcmp(curr->key, key)) {
        curr = curr->next;
    }

    *entry = curr;
    htable_errno = (curr) ? HTABLE_SUCCESS : HTABLE_ENOVAL;

  fn_exit:
    return htable_errno;
}
#endif /* End of __HTABLE_H__ */
