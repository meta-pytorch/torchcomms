/**
 * @file    amdsmi_hello.c
 * @author  Dong Jun Woun <djwoun@gmail.com>
 * @brief   Minimal example that reads a single AMD-SMI event via PAPI's AMD-SMI component.
 * @details Selects the event from argv[1] if provided; otherwise chooses the
 *          first native AMD-SMI event by enumerating the component (like amdsmi_basics.c).
 *          Requires PAPI_AMDSMI_ROOT so the component can dlopen the AMD-SMI library.
 *          Uses the test harness (test_harness.h) for consistent output and skip handling.
 */

#include "test_harness.h"

#include "papi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // Disable stdout buffering so the harness status line appears immediately.
    setvbuf(stdout, NULL, _IONBF, 0);

    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);

    // Selected event: argv[1] if provided, else we'll enumerate and pick the first AMD-SMI native event.
    char ev_buf[PAPI_MAX_STR_LEN] = {0};
    const char* ev = NULL;
    if (argc > 1 && strncmp(argv[1], "--", 2) != 0) {
        if (harness_canonicalize_event_name(argv[1], ev_buf, sizeof(ev_buf)) == PAPI_OK) {
            ev = ev_buf;
        } else {
            ev = argv[1];
        }
    }

    // Check AMD-SMI root so the component can dlopen the library.
    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) {
        SKIP("PAPI_AMDSMI_ROOT not set");
    }

    // Initialize PAPI.
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        NOTE("PAPI_library_init failed: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // If no event was passed, enumerate the AMD-SMI component and pick the first event (like amdsmi_basics.c).
    if (!ev) {
        int cid = -1;
        const int ncomps = PAPI_num_components();
        for (int i = 0; i < ncomps && cid < 0; ++i) {
            const PAPI_component_info_t *cinfo = PAPI_get_component_info(i);
            if (cinfo && strcmp(cinfo->name, "amd_smi") == 0) {
                cid = i;
            }
        }
        if (cid < 0) {
            SKIP("Unable to locate the amd_smi component (PAPI built without ROCm?)");
        }

        int base_code = PAPI_NATIVE_MASK;
        if (PAPI_enum_cmp_event(&base_code, PAPI_ENUM_FIRST, cid) != PAPI_OK) {
            SKIP("No native events found for AMD-SMI component");
        }

        char base_name[PAPI_MAX_STR_LEN] = {0};
        if (PAPI_event_code_to_name(base_code, base_name) != PAPI_OK || base_name[0] == '\0') {
            SKIP("Could not resolve AMD-SMI event name");
        }

        int qualified_code = 0;
        if (PAPI_event_name_to_code(base_name, &qualified_code) != PAPI_OK) {
            SKIP("Could not canonicalize AMD-SMI event name");
        }

        if (PAPI_event_code_to_name(qualified_code, ev_buf) != PAPI_OK || ev_buf[0] == '\0') {
            SKIP("Could not resolve canonical AMD-SMI event name");
        }

        ev = ev_buf;
        NOTE("Defaulting to first AMD-SMI event: %s", ev);
    }

    // Create an EventSet.
    int EventSet = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_create_eventset: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // Add the selected event.
    papi_errno = PAPI_add_named_event(EventSet, ev);
    if (papi_errno == PAPI_ENOEVNT || papi_errno == PAPI_ECNFLCT ||
        papi_errno == PAPI_EPERM) {
        NOTE("Event unavailable or HW/resource-limited: %s (%s)", ev,
             PAPI_strerror(papi_errno));
        SKIP("Event unavailable or HW/resource-limited");
    } else if (papi_errno != PAPI_OK) {
        NOTE("PAPI_add_named_event(%s): %s", ev, PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    // Start counters -> short wait -> stop/read.
    papi_errno = PAPI_start(EventSet);
    if (papi_errno == PAPI_ECNFLCT || papi_errno == PAPI_EPERM) {
        NOTE("Cannot start counters: %s", PAPI_strerror(papi_errno));
        SKIP("Cannot start counters");
    } else if (papi_errno != PAPI_OK) {
        NOTE("PAPI_start: %s", PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    usleep(100000); // ~100 ms sampling interval for this simple demo.

    long long val = 0;
    papi_errno = PAPI_stop(EventSet, &val);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_stop: %s", PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    (void)PAPI_cleanup_eventset(EventSet);
    (void)PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    // If --print was requested via the harness, emit the event name and value.
    if (opts.print) {
        printf("Event: %s\nValue: %lld\n", ev, val);
    }

    return eval_result(opts, 0);
}
