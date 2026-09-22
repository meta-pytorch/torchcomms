/**
 * @file    amdsmi_set_test.c
 * @brief   Exercise AMD SMI writable controls (power cap, fan speed) via PAPI.
 */

#include "test_harness.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int eventset;
    int code;
    char name[PAPI_MAX_STR_LEN];
} event_handle_t;

static int canonicalize_event_name(const char *input,
                                   char *output,
                                   size_t len) {
    return harness_canonicalize_event_name(input, output, len);
}

static int prepare_event_handle(int cid, const char *event_name,
                                event_handle_t *handle) {
    if (!handle)
        return PAPI_EINVAL;

    memset(handle, 0, sizeof(*handle));
    handle->eventset = PAPI_NULL;

    int rc = canonicalize_event_name(event_name, handle->name,
                                     sizeof(handle->name));
    if (rc != PAPI_OK)
        return rc;

    rc = PAPI_create_eventset(&handle->eventset);
    if (rc != PAPI_OK)
        return rc;

    rc = PAPI_assign_eventset_component(handle->eventset, cid);
    if (rc != PAPI_OK)
        goto fail;

    rc = PAPI_event_name_to_code(handle->name, &handle->code);
    if (rc != PAPI_OK)
        goto fail;

    rc = PAPI_add_event(handle->eventset, handle->code);
    if (rc != PAPI_OK)
        goto fail;

    return PAPI_OK;

fail:
    if (handle->eventset != PAPI_NULL) {
        (void)PAPI_cleanup_eventset(handle->eventset);
        (void)PAPI_destroy_eventset(&handle->eventset);
        handle->eventset = PAPI_NULL;
    }
    return rc;
}

static void destroy_event_handle(event_handle_t *handle) {
    if (!handle)
        return;
    if (handle->eventset != PAPI_NULL) {
        (void)PAPI_cleanup_eventset(handle->eventset);
        (void)PAPI_destroy_eventset(&handle->eventset);
        handle->eventset = PAPI_NULL;
    }
}

static int read_scalar_event(int cid, const char *event_name, long long *out) {
    if (!out)
        return PAPI_EINVAL;

    event_handle_t handle;
    int rc = prepare_event_handle(cid, event_name, &handle);
    if (rc != PAPI_OK)
        return rc;

    rc = PAPI_start(handle.eventset);
    if (rc != PAPI_OK) {
        destroy_event_handle(&handle);
        return rc;
    }

    long long value = 0;
    rc = PAPI_read(handle.eventset, &value);
    long long stop_val = 0;
    (void)PAPI_stop(handle.eventset, &stop_val);
    destroy_event_handle(&handle);

    if (rc == PAPI_OK)
        *out = value;
    return rc;
}

static long long clamp_long_long(long long value, long long min_value, long long max_value) {
    if (value < min_value)
        return min_value;
    if (value > max_value)
        return max_value;
    return value;
}

static void log_papi_failure(const char *context, int rc) {
    WARNF("%s failed: %s", context, PAPI_strerror(rc));
}

static void test_power_cap(int cid, HarnessOpts opts) {
    (void)opts;

    event_handle_t handle;
    int rc = prepare_event_handle(cid, "amd_smi:::power_cap", &handle);
    if (rc != PAPI_OK) {
        EXIT_WARNING_ON_ADD(rc, "amd_smi:::power_cap");
        return;
    }

    long long min_cap = 0;
    rc = read_scalar_event(cid, "amd_smi:::power_cap_range_min", &min_cap);
    if (rc != PAPI_OK)
        min_cap = 0;

    long long max_cap = 0;
    rc = read_scalar_event(cid, "amd_smi:::power_cap_range_max", &max_cap);
    if (rc != PAPI_OK)
        max_cap = 0;

    rc = PAPI_start(handle.eventset);
    if (rc != PAPI_OK) {
        EXIT_WARNING_ON_START(rc, handle.name);
        destroy_event_handle(&handle);
        return;
    }

    long long original = 0;
    rc = PAPI_read(handle.eventset, &original);
    if (rc != PAPI_OK) {
        WARNF("Failed to read %s: %s", handle.name, PAPI_strerror(rc));
        goto done;
    }

    long long target = original;
    long long delta = original / 20;
    if (delta <= 0)
        delta = 1;

    if (min_cap > 0 && max_cap > min_cap) {
        target = original - delta;
        if (target < min_cap)
            target = min_cap;
        if (target == original && max_cap > original)
            target = original + delta;
        if (target > max_cap)
            target = max_cap;
    } else {
        if (original > delta)
            target = original - delta;
        else if (max_cap > original)
            target = original + delta;
    }

    if (target == original) {
        WARNF("power_cap: unable to choose alternate value (orig=%lld)", original);
        goto done;
    }

    rc = PAPI_write(handle.eventset, &target);
    if (rc != PAPI_OK) {
        WARNF("power_cap write failed: %s", PAPI_strerror(rc));
        goto done;
    }

    long long verify = 0;
    if (PAPI_read(handle.eventset, &verify) == PAPI_OK)
        NOTE("power_cap changed from %lld to %lld", original, verify);

    rc = PAPI_write(handle.eventset, &original);
    if (rc != PAPI_OK)
        WARNF("power_cap restore failed: %s", PAPI_strerror(rc));
    else {
        long long restored = 0;
        if (PAPI_read(handle.eventset, &restored) == PAPI_OK)
            NOTE("power_cap restored to %lld", restored);
    }

  done:
    {
        long long stop_val = 0;
        (void)PAPI_stop(handle.eventset, &stop_val);
    }
    destroy_event_handle(&handle);
}

static void test_fan_speed(int cid, HarnessOpts opts) {
    (void)opts;

    event_handle_t handle;
    int rc = prepare_event_handle(cid, "amd_smi:::fan_speed_sensor=0", &handle);
    if (rc != PAPI_OK) {
        if (rc == PAPI_ENOEVNT)
            NOTE("Skipping fan_speed test: event unavailable");
        else
            EXIT_WARNING_ON_ADD(rc, "amd_smi:::fan_speed_sensor=0");
        return;
    }

    long long max_speed = 255;
    if (read_scalar_event(cid, "amd_smi:::fan_speed_max_sensor=0", &max_speed) != PAPI_OK ||
        max_speed <= 0)
        max_speed = 255;

    rc = PAPI_start(handle.eventset);
    if (rc != PAPI_OK) {
        if (rc == PAPI_ENOEVNT)
            NOTE("Skipping fan_speed test: start unavailable (ENOEVNT)");
        else
            EXIT_WARNING_ON_START(rc, handle.name);
        destroy_event_handle(&handle);
        return;
    }

    long long original = 0;
    rc = PAPI_read(handle.eventset, &original);
    if (rc != PAPI_OK) {
        log_papi_failure("PAPI_read(fan_speed)", rc);
        goto done;
    }

    long long delta = max_speed / 20;
    if (delta <= 0)
        delta = 1;

    long long target = clamp_long_long(original - delta, 0, max_speed);
    if (target == original)
        target = clamp_long_long(original + delta, 0, max_speed);

    if (target == original) {
        WARNF("fan_speed: unable to pick alternate value from %lld", original);
        goto done;
    }

    rc = PAPI_write(handle.eventset, &target);
    if (rc != PAPI_OK) {
        if (rc == PAPI_EPERM || rc == PAPI_ENOSUPP || rc == PAPI_ENOEVNT)
            WARNF("fan_speed write unavailable (%s)", PAPI_strerror(rc));
        else
            log_papi_failure("PAPI_write(fan_speed)", rc);
        goto done;
    }

    long long verify = 0;
    if (PAPI_read(handle.eventset, &verify) == PAPI_OK)
        NOTE("fan_speed now %lld (requested %lld)", verify, target);

    rc = PAPI_write(handle.eventset, &original);
    if (rc != PAPI_OK)
        log_papi_failure("PAPI_write(fan_speed restore)", rc);

  done:
    {
        long long stop_val = 0;
        (void)PAPI_stop(handle.eventset, &stop_val);
    }
    destroy_event_handle(&handle);
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);

    int rc = PAPI_library_init(PAPI_VER_CURRENT);
    if (rc != PAPI_VER_CURRENT) {
        WARNF("PAPI_library_init failed: %s", PAPI_strerror(rc));
        return eval_result(opts, 1);
    }

    int cid = -1;
    int num_comps = PAPI_num_components();
    for (int i = 0; i < num_comps; ++i) {
        const PAPI_component_info_t *cinfo = PAPI_get_component_info(i);
        if (cinfo && strcmp(cinfo->name, "amd_smi") == 0) {
            cid = i;
            break;
        }
    }

    if (cid < 0)
        SKIP("AMD SMI component not available");

    test_power_cap(cid, opts);
    test_fan_speed(cid, opts);

    PAPI_shutdown();
    return eval_result(opts, 0);
}
