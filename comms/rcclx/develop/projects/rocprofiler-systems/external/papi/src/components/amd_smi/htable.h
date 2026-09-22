/**
 * @file    htable.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#ifndef __HTABLE_H__
#define __HTABLE_H__

#include <string.h>
#include <inttypes.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/* Return codes for hash table operations */
#define HTABLE_SUCCESS   0    /* Operation successful */
#define HTABLE_ENOVAL   -1    /* Key not found in table */
#define HTABLE_EINVAL   -2    /* Invalid argument (e.g., NULL handle or key) */
#define HTABLE_ENOMEM   -3    /* Allocation failure */

#define HTABLE_MIN_SIZE        8                       /* Minimum number of buckets */
#define HTABLE_NEEDS_TO_GROW(table)   ((table)->size > 0 && (table)->capacity / (table)->size < 2)
#define HTABLE_NEEDS_TO_SHRINK(table) ((table)->size > 0 && (table)->capacity / (table)->size > 8)

/* Hash table entry definition (separate chaining) */
struct hash_table_entry {
    char *key;                      /* Dynamically allocated key string */
    void *val;                      /* Value associated with the key */
    struct hash_table_entry *next;  /* Next entry in the bucket's linked list */
};

/* Hash table structure */
struct hash_table {
    uint32_t capacity;              /* Number of buckets (table size) */
    uint32_t size;                  /* Number of entries currently stored */
    struct hash_table_entry **buckets; /* Array of bucket heads for separate chaining */
};

/* Internal helper function prototypes (not part of public API) */
static uint64_t hash_func(const char *str);
static int create_table(uint64_t capacity, struct hash_table **table);
static int destroy_table(struct hash_table *table);
static int rehash_table(struct hash_table *old_table, struct hash_table *new_table);
static int destroy_table_entries(struct hash_table *table);
static int create_table_entry(const char *key, void *val, struct hash_table_entry **entry);
static int destroy_table_entry(struct hash_table_entry *entry);
static int insert_table_entry(struct hash_table *table, struct hash_table_entry *entry);
static int delete_table_entry(struct hash_table *table, struct hash_table_entry *entry);
static int find_table_entry(struct hash_table *table, const char *key, struct hash_table_entry **entry);

/* Initialize a new hash table. Handle is an out-parameter for the table pointer. */
static inline int htable_init(void **handle)
{
    if (handle == NULL) {
        return HTABLE_EINVAL;
    }
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *table = NULL;
    /* Create initial table with minimum capacity */
    htable_errno = create_table(HTABLE_MIN_SIZE, &table);
    if (htable_errno != HTABLE_SUCCESS) {
        *handle = NULL;
        return htable_errno;
    }
    *handle = table;
    return HTABLE_SUCCESS;
}

/* Shutdown an existing hash table, freeing all allocated memory. */
static inline int htable_shutdown(void *handle)
{
    struct hash_table *table = (struct hash_table *) handle;
    if (table == NULL) {
        return HTABLE_EINVAL;
    }
    /* Free all entries and the table itself */
    destroy_table_entries(table);
    destroy_table(table);
    return HTABLE_SUCCESS;
}

/* Insert a key-value pair into the hash table. Updates value if key already exists. */
static inline int htable_insert(void *handle, const char *key, void *in)
{
    struct hash_table *table = (struct hash_table *) handle;
    if (table == NULL || key == NULL) {
        return HTABLE_EINVAL;
    }
    int htable_errno;
    struct hash_table_entry *entry = NULL;
    /* Check if key already exists */
    htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno == HTABLE_SUCCESS) {
        /* Key exists: update its value */
        entry->val = in;
        return HTABLE_SUCCESS;
    }
    /* Key not found: create a new entry */
    htable_errno = create_table_entry(key, in, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }
    /* Link the new entry into the table */
    htable_errno = insert_table_entry(table, entry);
    if (htable_errno != HTABLE_SUCCESS) {
        /* Insertion failed: free the entry and return error */
        papi_free(entry->key);
        papi_free(entry);
        return htable_errno;
    }
    /* Check if rehash (grow table) is needed after insertion */
    htable_errno = rehash_table(table, NULL);  /* use NULL to indicate self-resize (growth) */
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }
    return HTABLE_SUCCESS;
}

/* Remove an entry by key from the hash table. No effect if key not found. */
static inline int htable_delete(void *handle, const char *key)
{
    struct hash_table *table = (struct hash_table *) handle;
    if (table == NULL || key == NULL) {
        return HTABLE_EINVAL;
    }
    struct hash_table_entry *entry = NULL;
    int htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        /* Key not found or other error */
        return htable_errno;
    }
    /* Unlink the entry from the table (does not free memory yet) */
    htable_errno = delete_table_entry(table, entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }
    /* Free the removed entry structure */
    htable_errno = destroy_table_entry(entry);
    if (htable_errno != HTABLE_SUCCESS) {
        return htable_errno;
    }
    /* Check if rehash (shrink table) is needed after deletion */
    htable_errno = rehash_table(table, NULL);  /* attempt shrink after deletion */
    if (htable_errno == HTABLE_ENOMEM) {
        return htable_errno;
    }
    return htable_errno;
}

/* Find an entry by key in the hash table. 
 * Returns HTABLE_SUCCESS and sets *out if found, else HTABLE_ENOVAL. */
static inline int htable_find(void *handle, const char *key, void **out)
{
    struct hash_table *table = (struct hash_table *) handle;
    if (table == NULL || key == NULL || out == NULL) {
        return HTABLE_EINVAL;
    }
    struct hash_table_entry *entry = NULL;
    int htable_errno = find_table_entry(table, key, &entry);
    if (htable_errno != HTABLE_SUCCESS) {
        *out = NULL;  /* ensure output is NULL if not found */
        return htable_errno;
    }
    *out = entry->val;
    return HTABLE_SUCCESS;
}

/* djb2 string hash function – returns a 64-bit hash for the given string */
static uint64_t hash_func(const char *str)
{
    uint64_t hash = 5381ULL;
    int c;
    while ((c = *str++) != 0) {
        hash = ((hash << 5) + hash) + (uint8_t)c;  /* hash * 33 + c */
    }
    return hash;
}

/* Allocate and initialize a new hash_table structure with the given capacity. */
static int create_table(uint64_t capacity, struct hash_table **table)
{
    if (capacity < 1 || table == NULL) {
        return HTABLE_EINVAL;
    }
    int htable_errno = HTABLE_SUCCESS;
    struct hash_table *t = papi_calloc(1, sizeof(struct hash_table));
    if (t == NULL) {
        return HTABLE_ENOMEM;
    }
    t->buckets =  papi_calloc(capacity, sizeof(struct hash_table_entry *));
    if (t->buckets == NULL) {
        papi_free(t);
        return HTABLE_ENOMEM;
    }
    t->capacity = (uint32_t) capacity;
    t->size = 0;
    *table = t;
    return HTABLE_SUCCESS;
}

/* Free the memory associated with a hash_table (structure and bucket array). */
static int destroy_table(struct hash_table *table)
{
    if (table == NULL) {
        return HTABLE_SUCCESS;
    }
    if (table->buckets != NULL) {
        papi_free(table->buckets);
    }
    papi_free(table);
    return HTABLE_SUCCESS;
}

/* Rehash the entries from old_table into new_table or perform in-place resizing.
   If new_table is NULL, this function checks old_table and resizes it if needed. */
static int rehash_table(struct hash_table *old_table, struct hash_table *new_table)
{
    int htable_errno = HTABLE_SUCCESS;
    if (new_table == NULL) {
        /* Self-resizing mode: determine if growth or shrink is needed */
        char resize = (HTABLE_NEEDS_TO_GROW(old_table) << 1) | HTABLE_NEEDS_TO_SHRINK(old_table);
        if (!resize) {
            return HTABLE_SUCCESS;  /* no resizing needed */
        }
        /* Determine new capacity (double or half) */
        uint64_t new_capacity = (resize & 0x2) ? 
            (uint64_t)old_table->capacity * 2 
            : (uint64_t)old_table->capacity / 2;
        if (new_capacity < HTABLE_MIN_SIZE) {
            new_capacity = HTABLE_MIN_SIZE;
        }
        /* Allocate a new table structure and buckets */
        htable_errno = create_table(new_capacity, &new_table);
        if (htable_errno != HTABLE_SUCCESS) {
            return htable_errno;
        }
        /* Move all entries from old_table into new_table */
        for (uint64_t i = 0; i < old_table->capacity; ++i) {
            struct hash_table_entry *entry = old_table->buckets[i];
            while (entry != NULL) {
                struct hash_table_entry *next_entry = entry->next;
                /* Compute new bucket index (capacity is always power-of-2) */
                uint64_t new_index = hash_func(entry->key) & (new_table->capacity - 1);
                /* Insert entry at head of new_table's bucket list */
                entry->next = new_table->buckets[new_index];
                new_table->buckets[new_index] = entry;
                entry = next_entry;
            }
        }
        new_table->size = old_table->size;
        /* Replace old_table's data with new_table's data */
        struct hash_table_entry **old_buckets = old_table->buckets;
        old_table->capacity = new_table->capacity;
        old_table->size = new_table->size;
        old_table->buckets = new_table->buckets;
        new_table->buckets = NULL;  /* avoid double-free */
        /* Free old bucket array and temporary table structure */
        papi_free(old_buckets);
        destroy_table(new_table);
        return HTABLE_SUCCESS;
    }
    /* Explicit rehash into a provided new_table (for manual resizing, if needed) */
    for (uint64_t j = 0; j < old_table->capacity; ++j) {
        struct hash_table_entry *entry = old_table->buckets[j];
        while (entry != NULL) {
            struct hash_table_entry *next_entry = entry->next;
            uint64_t new_index = hash_func(entry->key) & (new_table->capacity - 1);
            entry->next = new_table->buckets[new_index];
            new_table->buckets[new_index] = entry;
            entry = next_entry;
        }
    }
    new_table->size = old_table->size;
    return HTABLE_SUCCESS;
}

/* Free all entries in the hash table (but not the table or buckets themselves). */
static int destroy_table_entries(struct hash_table *table)
{
    if (table == NULL) {
        return HTABLE_SUCCESS;
    }
    for (uint64_t i = 0; i < table->capacity; ++i) {
        struct hash_table_entry *entry = table->buckets[i];
        while (entry != NULL) {
            struct hash_table_entry *tmp = entry;
            entry = entry->next;
            papi_free(tmp->key);
            papi_free(tmp);
        }
        table->buckets[i] = NULL;
    }
    table->size = 0;
    return HTABLE_SUCCESS;
}

/* Create a new hash_table_entry with the given key and value. Copies the key string. */
static int create_table_entry(const char *key, void *val, struct hash_table_entry **entry)
{
    if (key == NULL || entry == NULL) {
        return HTABLE_EINVAL;
    }
    struct hash_table_entry *e = papi_calloc(1, sizeof(struct hash_table_entry));
    if (e == NULL) {
        return HTABLE_ENOMEM;
    }
    e->key = papi_strdup(key);
    if (e->key == NULL) {  /* strdup failure */
        papi_free(e);
        return HTABLE_ENOMEM;
    }
    e->val = val;
    e->next = NULL;
    *entry = e;
    return HTABLE_SUCCESS;
}

/* Destroy a single hash_table_entry (free its key and memory). */
static int destroy_table_entry(struct hash_table_entry *entry)
{
    if (entry == NULL) {
        return HTABLE_EINVAL;
    }
    papi_free(entry->key);
    papi_free(entry);
    return HTABLE_SUCCESS;
}

/* Insert a hash_table_entry into the table (at the head of its bucket list). */
static int insert_table_entry(struct hash_table *table, struct hash_table_entry *entry)
{
    if (table == NULL || entry == NULL) {
        return HTABLE_EINVAL;
    }
    /* Compute bucket index and insert at head of list */
    uint64_t index = hash_func(entry->key) & (table->capacity - 1);
    entry->next = table->buckets[index];
    table->buckets[index] = entry;
    table->size += 1;
    return HTABLE_SUCCESS;
}

/* Remove a hash_table_entry from its bucket list (does not free the entry). */
static int delete_table_entry(struct hash_table *table, struct hash_table_entry *entry)
{
    if (table == NULL || entry == NULL) {
        return HTABLE_EINVAL;
    }
    uint64_t index = hash_func(entry->key) & (table->capacity - 1);
    struct hash_table_entry *curr = table->buckets[index];
    struct hash_table_entry *prev = NULL;
    while (curr != NULL) {
        if (curr == entry) {
            /* Found the entry to remove */
            if (prev == NULL) {
                /* Entry is at head of the list */
                table->buckets[index] = curr->next;
            } else {
                /* Entry is in the middle or end of the list */
                prev->next = curr->next;
            }
            entry->next = NULL;
            table->size -= 1;
            return HTABLE_SUCCESS;
        }
        prev = curr;
        curr = curr->next;
    }
    /* Entry not found (should not happen if a valid pointer was provided) */
    return HTABLE_ENOVAL;
}

/* Find a hash_table_entry by key in the table. Sets *entry if found. */
static int find_table_entry(struct hash_table *table, const char *key, struct hash_table_entry **entry)
{
    if (table == NULL || key == NULL || entry == NULL) {
        return HTABLE_EINVAL;
    }
    uint64_t index = hash_func(key) & (table->capacity - 1);
    struct hash_table_entry *curr = table->buckets[index];
    while (curr != NULL && strcmp(curr->key, key) != 0) {
        curr = curr->next;
    }
    *entry = curr;
    return (curr != NULL ? HTABLE_SUCCESS : HTABLE_ENOVAL);
}

#endif /* __HTABLE_H__ */
