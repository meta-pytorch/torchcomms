/**
 * @file    amdsmi_energy_monotonic.c
 * @author  Dong Jun Woun
 *          djwoun@gmail.com
 * @brief   Verifies that the AMD SMI energy counter exposed via PAPI increases
 *          monotonically by sampling twice about one second apart.
 *
 * @details This small harnessed test:
 *   1) Ensures PAPI + AMD-SMI are available (via PAPI_AMDSMI_ROOT).
 *   2) Adds the "amd_smi:::energy_consumed:device=0" event to an event set.
 *   3) Starts counting, reads once, then polls for up to ~1s for an increase.
 *   4) Reports PASS if the second sample is greater than the first.
 *
 *   The NOTE/SKIP macros come from the project test harness.
 */

#include "test_harness.h"
#include "papi.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    // Parse common test harness options (quiet/print/exit codes, etc.).
    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);

    // Ensure the AMD-SMI PAPI component is configured.
    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) {
        SKIP("PAPI_AMDSMI_ROOT not set");
    }

    // Initialize the PAPI library.
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        NOTE("PAPI_library_init failed: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // Create an empty event set and add the AMD-SMI energy counter for device 0.
    int EventSet = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_create_eventset: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    const char *ev = "amd_smi:::energy_consumed:device=0";
    papi_errno = PAPI_add_named_event(EventSet, ev);
    if (papi_errno == PAPI_ENOEVNT) {
        SKIP("energy_consumed:device=0 not supported");
    } else if (papi_errno != PAPI_OK) {
        NOTE("PAPI_add_named_event(%s): %s", ev, PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // Begin counting.
    papi_errno = PAPI_start(EventSet);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_start: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    long long v1 = 0, v2 = 0;

    // First sample.
    papi_errno = PAPI_read(EventSet, &v1);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_read(1): %s", PAPI_strerror(papi_errno));
        long long dummy = 0; PAPI_stop(EventSet, &dummy);
        return eval_result(opts, 1);
    }

    // Poll for up to ~1 second for the energy counter to advance.
    for (int i = 0; i < 10; ++i) {
        usleep(100000); // 100 ms

        papi_errno = PAPI_read(EventSet, &v2);
        if (papi_errno != PAPI_OK) {
            NOTE("PAPI_read(2): %s", PAPI_strerror(papi_errno));
            long long dummy = 0; PAPI_stop(EventSet, &dummy);
            return eval_result(opts, 1);
        }
        if (v2 > v1) break; // monotonic increase observed
    }

    // Clean up PAPI resources.
    long long dummy = 0;
    PAPI_stop(EventSet, &dummy);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    if (opts.print) {
        printf("energy_consumed: first=%lld  second=%lld  delta=%lld\n",
               v1, v2, (v2 - v1));
    }

    // Fail if we never observed an increase.
    int failed = (v2 <= v1) ? 1 : 0;
    if (failed) NOTE("Energy did not increase");

    return eval_result(opts, failed);
}
