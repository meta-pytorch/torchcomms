#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "papi.h"
#include "papi_test.h"
#define MAX_LOCAL_RANKS (512)

#define MPI_CALL(call, err_handle) do { \
    int _status = (call);               \
    if (_status == MPI_SUCCESS)         \
        break;                          \
    err_handle;                         \
} while(0)

static int
cmp_fn(const void *a, const void *b)
{
    return (*(unsigned long *)a - *(unsigned long *)b);
}

static int
print_cuda_affinity( MPI_Comm comm, void *handle )
{
    int shm_comm_size;
    MPI_Comm shm_comm = MPI_COMM_NULL;
    const char *name;
    int dev_count;
    int rank, local_rank;
    int ranks[MAX_LOCAL_RANKS] = { 0 };
    unsigned long uid;
    unsigned long uids[MAX_LOCAL_RANKS] = { 0 };

    MPI_CALL(MPI_Comm_rank(comm, &rank), return _status);
    MPI_CALL(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,
                                 rank, MPI_INFO_NULL, &shm_comm),
             return _status);
    MPI_CALL(MPI_Comm_size(shm_comm, &shm_comm_size), return _status);
    MPI_CALL(MPI_Comm_rank(shm_comm, &local_rank), return _status);

    PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, &dev_count);
    int i;
    for (i = 0; i < dev_count; ++i) {
        PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME, &name);
        PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_ULONG_UID, &uid);

        MPI_CALL(MPI_Alltoall(&rank, 1, MPI_INT, ranks, 1, MPI_INT,
                              shm_comm),
                 return _status);
        MPI_CALL(MPI_Alltoall(&uid, 1, MPI_UNSIGNED_LONG, uids, 1,
                              MPI_UNSIGNED_LONG, shm_comm),
                 return _status);

        unsigned long sorted_uids[MAX_LOCAL_RANKS] = { 0 };
        unsigned long uniq_sorted_uids[MAX_LOCAL_RANKS] = { 0 };
        memcpy(sorted_uids, uids, sizeof(unsigned long) * shm_comm_size);
        qsort(sorted_uids, shm_comm_size, sizeof(unsigned long), cmp_fn);

        if (local_rank == 0) {
            int j, uniq_uids = 0;
            unsigned long curr_uid = 0;
            for (j = 0; j < shm_comm_size; ++j) {
                if (sorted_uids[j] != curr_uid) {
                    curr_uid = sorted_uids[j];
                    uniq_sorted_uids[uniq_uids++] = curr_uid;
                }
            } 

            int k, l, list[MAX_LOCAL_RANKS] = { 0 };
            for (j = 0, l = 0; j < uniq_uids; ++j) {
                for (k = 0; k < shm_comm_size; ++k) {
                    if (uids[k] == uniq_sorted_uids[j]) {
                        list[l++] = ranks[k];
                    }
                }

                printf( "GPU-%i Affinity                       : Name: %s, UID: %lu, Ranks: [ ",
                        i, name, uniq_sorted_uids[j] );
                for (k = 0; k < l; ++k) {
                    printf( "%d ", list[k] );
                }
                printf( "]\n" );
            }
        }
    }

    MPI_CALL(MPI_Comm_free(&shm_comm), return _status);
    return 0;
}

int main(int argc, char *argv[])
{
    int quiet = 0;
    quiet = tests_quiet(argc, argv);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", retval);
    }

    if (!quiet && rank == 0) {
        printf("Testing systedect component with PAPI %d.%d.%d\n",
               PAPI_VERSION_MAJOR(PAPI_VERSION),
               PAPI_VERSION_MINOR(PAPI_VERSION),
               PAPI_VERSION_REVISION(PAPI_VERSION));
    }

    void *handle;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__CUDA;

    while (PAPI_enum_dev_type(enum_modifier, &handle) == PAPI_OK) {
        if (!quiet) {
            print_cuda_affinity(MPI_COMM_WORLD, handle);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        test_pass(__FILE__);
    }

    MPI_Finalize();
    PAPI_shutdown();
    return 0;
}
