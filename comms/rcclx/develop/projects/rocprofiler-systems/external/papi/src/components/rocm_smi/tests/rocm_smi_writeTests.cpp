//-----------------------------------------------------------------------------
// This program must be compiled using a special makefile:
// make -f ROCM_SMI_Makefile rocm_smi_writeTests.out
//-----------------------------------------------------------------------------
#define __HIP_PLATFORM_HCC__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "papi.h"
#include <hip/hip_runtime.h>
#include "force_init.h"

// Helper Function
void write_papi_event(int cid, const char* event_name, long long value_to_write);
void read_and_print_current_values(int cid,
                                   const char* perf_name, 
                                   const char* pcap_name, 
                                   const char* fan_name,
                                   const char* pcap_max_name, 
                                   const char* fan_max_name,
                                   const char* stage_label);

#define CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// THIS MACRO EXITS if the papi call does not return PAPI_OK.
#define CALL_PAPI_OK(papi_routine) \
    do { \
        int _papiret = papi_routine; \
        if (_papiret != PAPI_OK) { \
            fprintf(stderr, "%s:%d macro: PAPI Error: function " #papi_routine " failed with ret=%d [%s].\n", \
                    __FILE__, __LINE__, _papiret, PAPI_strerror(_papiret)); \
            exit(-1); \
        } \
    } while (0);


// Show help.
//-----------------------------------------------------------------------------
static void printUsage()
{
    printf("Demonstrate use of ROCM API write routines.\n");
    printf("This program will use PAPI to read ROCm SMI values, attempt to write\n");
    printf("modified values for perf_level, power_cap, and fan_speed (for device 0),\n");
    printf("read them back, revert them to original values, and read again.\n");
    printf("Requires necessary permissions to write ROCm SMI values.\n");
    printf("Compile with: make -f ROCM_SMI_Makefile rocm_smi_writeTests.out\n");
}

//-----------------------------------------------------------------------------
// Interpret command line flags.
//-----------------------------------------------------------------------------
void parseCommandLineArgs(int argc, char *argv[])
{
    int i;
    for (i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--help") == 0) ||
            (strcmp(argv[i], "-help") == 0)  ||
            (strcmp(argv[i], "-h") == 0)) {
            printUsage();
            exit(0);
        }
    }
}

//-----------------------------------------------------------------------------
// Main program.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int devices;
    int i = 0;
    int r;

    parseCommandLineArgs(argc, argv);

    int ret;
    int k, cid = -1;

    // PAPI Initialization
    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed, ret=%i [%s]\n",
                ret, PAPI_strerror(ret));
        exit(-1);
    }
    printf("PAPI version: %d.%d.%d\n", 
           PAPI_VERSION_MAJOR(PAPI_VERSION), 
           PAPI_VERSION_MINOR(PAPI_VERSION), 
           PAPI_VERSION_REVISION(PAPI_VERSION));
    fflush(stdout);

    // Find rocm_smi component
    k = PAPI_num_components();
    for (i = 0; i < k && cid < 0; i++) {
        const PAPI_component_info_t *aComponent = PAPI_get_component_info(i);
        if (aComponent && strcmp("rocm_smi", aComponent->name) == 0) cid = i;
    }
    if (cid < 0) {
        fprintf(stderr, "Failed to find rocm_smi component.\n");
        PAPI_shutdown();
        exit(-1);
    }
    printf("Found ROCM_SMI Component at id %d\n", cid);

    // Force Init
    force_rocm_smi_init(cid);

    // Get Device Count
    {
        int tempEventSet = PAPI_NULL;
        long long numDevValue = 0;
        CALL_PAPI_OK(PAPI_create_eventset(&tempEventSet));
        CALL_PAPI_OK(PAPI_assign_eventset_component(tempEventSet, cid));
        ret = PAPI_add_named_event(tempEventSet, "rocm_smi:::NUMDevices");
        if (ret == PAPI_OK) {
            CALL_PAPI_OK(PAPI_start(tempEventSet));
            CALL_PAPI_OK(PAPI_stop(tempEventSet, &numDevValue));
            devices = (int)numDevValue;
            printf("Found %d devices.\n", devices);
        } else {
            fprintf(stderr, "FAILED to add NUMDevices event.\n");
            CALL_PAPI_OK(PAPI_cleanup_eventset(tempEventSet));
            CALL_PAPI_OK(PAPI_destroy_eventset(&tempEventSet));
            exit(-1);
        }
        CALL_PAPI_OK(PAPI_cleanup_eventset(tempEventSet));
        CALL_PAPI_OK(PAPI_destroy_eventset(&tempEventSet));
    }
    // Handle no devices
    if (devices < 1) {
        fprintf(stderr, "No ROCm devices found.\n"); 
        PAPI_shutdown(); 
        exit(0);
    }


    long long initial_perf_level = -1;
    long long initial_power_cap = -1;
    long long initial_fan_speed = -1;
    long long power_cap_range_max_val = -1;
    long long fan_speed_max_val = -1;
    char perf_level_event_name[PAPI_MAX_STR_LEN] = "";
    char power_cap_event_name[PAPI_MAX_STR_LEN] = "";
    char fan_speed_event_name[PAPI_MAX_STR_LEN] = "";
    char power_cap_range_max_event_name[PAPI_MAX_STR_LEN] = "";
    char fan_speed_max_event_name[PAPI_MAX_STR_LEN] = "";
    bool can_write_perf = false;
    bool can_write_pcap = false;
    bool can_write_fan = false;
    long long new_perf_level = -1;
    long long new_power_cap = -1;
    long long new_fan_speed = -1;

    // ---- Initial Read ----
    printf("\n--- Initial Read: Finding events and getting base values ---\n");
    const char* target_substrings[] = {
        "perf_level", "power_cap:", "power_cap_range_max", "fan_speed:", "fan_speed_max"
    };
    const int num_target_substrings = sizeof(target_substrings) / sizeof(target_substrings[0]);
    const int MAX_ROCM_EVENTS = 512;
    char event_names[MAX_ROCM_EVENTS][PAPI_MAX_STR_LEN];
    long long *rocm_values = NULL;
    int num_rocm_events = 0;
    int event_code = PAPI_NATIVE_MASK;
    char current_event_name[PAPI_MAX_STR_LEN];
    int readEventSet = PAPI_NULL;

    CALL_PAPI_OK(PAPI_create_eventset(&readEventSet));
    CALL_PAPI_OK(PAPI_assign_eventset_component(readEventSet, cid));

    printf("Enumerating events to find targets (device=0, sensor=0 where applicable) for initial read...\n");
    r = PAPI_enum_cmp_event(&event_code, PAPI_ENUM_FIRST, cid);
    while (r == PAPI_OK) {
        ret = PAPI_event_code_to_name(event_code, current_event_name);
        if (ret != PAPI_OK) { 
            fprintf(stderr, "Warning: PAPI_event_code_to_name failed for code %#x: %s\n", 
                   event_code, PAPI_strerror(ret)); 
            r = PAPI_enum_cmp_event(&event_code, PAPI_ENUM_EVENTS, cid); 
            continue; 
        }

        bool is_target = false;
        const char* matched_substring = NULL;
        for (i = 0; i < num_target_substrings; ++i) {
            if (strstr(current_event_name, target_substrings[i]) != NULL) {
                bool device_match = (strstr(current_event_name, ":device=0") != NULL);
                if (strcmp(target_substrings[i],"perf_level") == 0) {
                    if (device_match) { 
                        is_target = true; 
                        matched_substring = target_substrings[i]; 
                        break; 
                    }
                } else {
                    bool sensor_match = (strstr(current_event_name, ":sensor=0") != NULL);
                    if (device_match && sensor_match) { 
                        is_target = true; 
                        matched_substring = target_substrings[i]; 
                        break; 
                    }
                    else if (device_match && strstr(current_event_name, ":sensor=") == NULL){
                         if (strcmp(target_substrings[i],"power_cap:")==0 || 
                             strcmp(target_substrings[i],"fan_speed:")==0) {
                              printf("  Warning: Matched '%s' for device 0 but no sensor specified: %s\n", 
                                     target_substrings[i], current_event_name);
                              is_target = true; 
                              matched_substring = target_substrings[i]; 
                              break;
                         }
                    }
                }
            }
        }

        if (is_target) {
            if (num_rocm_events < MAX_ROCM_EVENTS) {
                ret = PAPI_add_event(readEventSet, event_code);
                if (ret == PAPI_OK) {
                    printf("  Adding event (matched '%s'): %s\n", matched_substring, current_event_name);
                    strncpy(event_names[num_rocm_events], current_event_name, PAPI_MAX_STR_LEN - 1);
                    event_names[num_rocm_events][PAPI_MAX_STR_LEN - 1] = '\0';
                    num_rocm_events++;
                } else { 
                    fprintf(stderr, "  Warning: Failed to add event %s: %s\n", 
                            current_event_name, PAPI_strerror(ret)); 
                    if(ret==PAPI_ENOMEM) break;
                }
            } else { 
                fprintf(stderr, "Error: Exceeded MAX_ROCM_EVENTS.\n"); 
                break; 
            }
        }
        r = PAPI_enum_cmp_event(&event_code, PAPI_ENUM_EVENTS, cid);
    }
    printf("Added %d events for initial read.\n", num_rocm_events);

    if (num_rocm_events > 0) {
        rocm_values = (long long *)calloc(num_rocm_events, sizeof(long long));
        if (!rocm_values) { /* Handle error */ exit(-1); }

        CALL_PAPI_OK(PAPI_start(readEventSet));
        CALL_PAPI_OK(PAPI_stop(readEventSet, rocm_values));

        printf("\n--- Extracting Initial Values and Event Names ---\n");
        for (i = 0; i < num_rocm_events; ++i) {
            printf("  Read Event %d: %-60s = %lld\n", i, event_names[i], rocm_values[i]);
            if (strstr(event_names[i], "power_cap_range_max") != NULL && 
                strstr(event_names[i], ":device=0") != NULL) {
                power_cap_range_max_val = rocm_values[i];
                strncpy(power_cap_range_max_event_name, event_names[i], PAPI_MAX_STR_LEN - 1);
                power_cap_range_max_event_name[PAPI_MAX_STR_LEN - 1] = '\0';
            } else if (strstr(event_names[i], "fan_speed_max") != NULL && 
                       strstr(event_names[i], ":device=0") != NULL) {
                fan_speed_max_val = rocm_values[i];
                strncpy(fan_speed_max_event_name, event_names[i], PAPI_MAX_STR_LEN - 1);
                fan_speed_max_event_name[PAPI_MAX_STR_LEN - 1] = '\0';
            } else if (strstr(event_names[i], "perf_level") != NULL && 
                       strstr(event_names[i], ":device=0") != NULL) {
                initial_perf_level = rocm_values[i];
                strncpy(perf_level_event_name, event_names[i], PAPI_MAX_STR_LEN - 1);
                perf_level_event_name[PAPI_MAX_STR_LEN - 1] = '\0';
            } else if (strstr(event_names[i], "power_cap:") != NULL && 
                       strstr(event_names[i], "power_cap_range_max") == NULL && 
                       strstr(event_names[i], ":device=0") != NULL) {
                initial_power_cap = rocm_values[i];
                strncpy(power_cap_event_name, event_names[i], PAPI_MAX_STR_LEN - 1);
                power_cap_event_name[PAPI_MAX_STR_LEN - 1] = '\0';
            } else if (strstr(event_names[i], "fan_speed:") != NULL && 
                       strstr(event_names[i], "fan_speed_max") == NULL && 
                       strstr(event_names[i], ":device=0") != NULL) {
                initial_fan_speed = rocm_values[i];
                strncpy(fan_speed_event_name, event_names[i], PAPI_MAX_STR_LEN - 1);
                fan_speed_event_name[PAPI_MAX_STR_LEN - 1] = '\0';
            }
        }
        free(rocm_values);
        rocm_values = NULL;
    } else {
        printf("No target events found for initial read. Skipping write tests.\n");
        goto cleanup_and_exit;
    }

    // Cleanup the initial read EventSet - Pass address to destroy
    CALL_PAPI_OK(PAPI_cleanup_eventset(readEventSet));
    CALL_PAPI_OK(PAPI_destroy_eventset(&readEventSet)); // Pass address
    readEventSet = PAPI_NULL;


    // ---- Stage 1: Calculate and Write NEW Values ----
    printf("\n=== Stage 1: Calculating and Writing NEW Values ===\n");
    can_write_perf = (initial_perf_level != -1 && strcmp(perf_level_event_name, "") != 0);
    can_write_pcap = (initial_power_cap != -1 && power_cap_range_max_val != -1 && strcmp(power_cap_event_name, "") != 0);
    can_write_fan = (initial_fan_speed != -1 && strcmp(fan_speed_event_name, "") != 0);

    if (can_write_perf) {
        new_perf_level = initial_perf_level + 1; // Example: Increment perf level
        printf("    Calculating new perf_level: %lld + 1 = %lld\n", initial_perf_level, new_perf_level);
        write_papi_event(cid, perf_level_event_name, new_perf_level);
    } else { 
        printf("Skipping perf_level write (initial value/name not found or invalid).\n"); 
    }

    if (can_write_pcap) {
        new_power_cap = power_cap_range_max_val - 1000000; // Example: 1W below max
        if (new_power_cap < 0) { new_power_cap = initial_power_cap; } // Basic sanity check
        printf("    Calculating new power_cap: %lld uW - 1000000 uW = %lld uW\n", 
               power_cap_range_max_val, new_power_cap);
        write_papi_event(cid, power_cap_event_name, new_power_cap);
    } else { 
        printf("Skipping power_cap write (initial value/name/max not found or invalid).\n"); 
    }

    if (can_write_fan) {
        new_fan_speed = fan_speed_max_val - 1; // Example: Decrease fan speed slightly
        if (new_fan_speed < 0) { new_fan_speed = 0; } // Basic sanity check (min speed 0?)
        printf("    Calculating new fan_speed: %lld - 1 = %lld\n", fan_speed_max_val, new_fan_speed);
        write_papi_event(cid, fan_speed_event_name, new_fan_speed);
    } else { 
        printf("Skipping fan_speed write (initial value/name not found or invalid).\n"); 
    }

    // ---- Stage 2: Read values AFTER writing NEW ones ----
    printf("\n=== Stage 2: Verifying NEW Values ===\n");
    read_and_print_current_values(cid, 
                                  perf_level_event_name, 
                                  power_cap_event_name, 
                                  fan_speed_event_name,
                                  power_cap_range_max_event_name, 
                                  fan_speed_max_event_name,
                                  "After Writing New Values");

    // ---- Stage 3: Write INITIAL values back (Revert) ----
    printf("\n=== Stage 3: Reverting to INITIAL Values ===\n");
    if (can_write_perf) { 
        write_papi_event(cid, perf_level_event_name, initial_perf_level); 
    } else { 
        printf("Skipping perf_level revert.\n"); 
    }
    
    if (can_write_pcap) { 
        write_papi_event(cid, power_cap_event_name, initial_power_cap); 
    } else { 
        printf("Skipping power_cap revert.\n"); 
    }
    
    if (can_write_fan) { 
        write_papi_event(cid, fan_speed_event_name, initial_fan_speed); 
    } else { 
        printf("Skipping fan_speed revert.\n"); 
    }

    // ---- Stage 4: Read values AFTER reverting ----
    printf("\n=== Stage 4: Verifying REVERTED Values ===\n");
    read_and_print_current_values(cid, 
                                  perf_level_event_name, 
                                  power_cap_event_name, 
                                  fan_speed_event_name,
                                  power_cap_range_max_event_name, 
                                  fan_speed_max_event_name,
                                  "After Reverting to Initial Values");

    // ---- Cleanup and Exit ----
cleanup_and_exit:
    printf("\n--- Write/Revert Test Sequence Finished ---\n");
    if (readEventSet != PAPI_NULL) { // Check if cleanup needed after jump
         printf("Performing cleanup for initial read EventSet after jump...\n");
         CALL_PAPI_OK(PAPI_cleanup_eventset(readEventSet));
         CALL_PAPI_OK(PAPI_destroy_eventset(&readEventSet)); // Pass address
    }
    printf("Finished All Tests.\n");
    PAPI_shutdown();
    return(0);
} // end MAIN.

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++ Helper Function Definitions ++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//-----------------------------------------------------------------------------
// Helper to write a single value to a specific PAPI event name. C version.
//-----------------------------------------------------------------------------
void write_papi_event(int cid, const char* event_name, long long value_to_write) {
    printf("    Attempting Write: Set '%s' = %lld\n", event_name, value_to_write);
    if (event_name == NULL || strcmp(event_name, "") == 0) { /* Handle error */ return; }

    int writeEventSet = PAPI_NULL;
    int ret;
    long long read_back_value;
    long long write_buffer[1];
    write_buffer[0] = value_to_write;

    CALL_PAPI_OK(PAPI_create_eventset(&writeEventSet)); // Pass address
    CALL_PAPI_OK(PAPI_assign_eventset_component(writeEventSet, cid));

    ret = PAPI_add_named_event(writeEventSet, event_name);
    if (ret != PAPI_OK) {
        fprintf(stderr, "    Error: FAILED to add event '%s' for writing, ret=%d [%s]. Skipping write.\n", 
                event_name, ret, PAPI_strerror(ret));
        CALL_PAPI_OK(PAPI_cleanup_eventset(writeEventSet));
        CALL_PAPI_OK(PAPI_destroy_eventset(&writeEventSet)); // Pass address
        return;
    }

    CALL_PAPI_OK(PAPI_start(writeEventSet));

    ret = PAPI_write(writeEventSet, write_buffer);
    if (ret != PAPI_OK) {
        fprintf(stderr, "    Error: PAPI_write FAILED for event '%s' with value %lld, ret=%d [%s].\n", 
                event_name, value_to_write, ret, PAPI_strerror(ret));
        int stop_ret = PAPI_stop(writeEventSet, &read_back_value);
        if (stop_ret != PAPI_OK) { 
            fprintf(stderr, "    Warning: PAPI_stop after failed PAPI_write also failed: %s\n", 
                    PAPI_strerror(stop_ret)); 
        }
    } else {
        printf("    PAPI_write call succeeded for '%s' = %lld.\n", event_name, value_to_write);
        CALL_PAPI_OK(PAPI_stop(writeEventSet, &read_back_value));
        printf("    Read back value immediately after write: %lld\n", read_back_value);
        if (read_back_value != value_to_write) { 
            printf("    Warning: Read-back value (%lld) does not match written value (%lld).\n", 
                   read_back_value, value_to_write); 
        }
    }

    CALL_PAPI_OK(PAPI_cleanup_eventset(writeEventSet));
    CALL_PAPI_OK(PAPI_destroy_eventset(&writeEventSet)); // Pass address
    printf("    Write attempt finished for '%s'.\n", event_name);
}

//-----------------------------------------------------------------------------
// Helper to read the set of relevant metrics (passed by name) and print them. C version.
//-----------------------------------------------------------------------------
#define MAX_EVENTS_TO_READ 10 // Max number of events this function will read at once

void read_and_print_current_values(int cid,
                                   const char* perf_name, 
                                   const char* pcap_name, 
                                   const char* fan_name,
                                   const char* pcap_max_name, 
                                   const char* fan_max_name,
                                   const char* stage_label)
{
    printf("    Reading Values [%s] for Verification <--\n", stage_label);

    int readSet = PAPI_NULL;
    int ret;
    char events_to_read[MAX_EVENTS_TO_READ][PAPI_MAX_STR_LEN];
    char event_short_names[MAX_EVENTS_TO_READ][50];
    bool added_flags[MAX_EVENTS_TO_READ];
    int read_count = 0;
    int i;

    memset(events_to_read, 0, sizeof(events_to_read));
    memset(event_short_names, 0, sizeof(event_short_names));
    for(i=0; i<MAX_EVENTS_TO_READ; ++i) added_flags[i] = false;

    if (perf_name && strcmp(perf_name, "") != 0 && read_count < MAX_EVENTS_TO_READ) { 
        strncpy(events_to_read[read_count], perf_name, PAPI_MAX_STR_LEN - 1); 
        strncpy(event_short_names[read_count], "Perf Level", 49); 
        read_count++; 
    }
    if (pcap_name && strcmp(pcap_name, "") != 0 && read_count < MAX_EVENTS_TO_READ) { 
        strncpy(events_to_read[read_count], pcap_name, PAPI_MAX_STR_LEN - 1); 
        strncpy(event_short_names[read_count], "Power Cap (uW)", 49); 
        read_count++; 
    }
    if (fan_name && strcmp(fan_name, "") != 0 && read_count < MAX_EVENTS_TO_READ) { 
        strncpy(events_to_read[read_count], fan_name, PAPI_MAX_STR_LEN - 1); 
        strncpy(event_short_names[read_count], "Fan Speed", 49); 
        read_count++; 
    }
    if (pcap_max_name && strcmp(pcap_max_name, "") != 0 && read_count < MAX_EVENTS_TO_READ) { 
        strncpy(events_to_read[read_count], pcap_max_name, PAPI_MAX_STR_LEN - 1); 
        strncpy(event_short_names[read_count], "Power Cap Max (uW)", 49); 
        read_count++; 
    }
    if (fan_max_name && strcmp(fan_max_name, "") != 0 && read_count < MAX_EVENTS_TO_READ) { 
        strncpy(events_to_read[read_count], fan_max_name, PAPI_MAX_STR_LEN - 1); 
        strncpy(event_short_names[read_count], "Fan Speed Max", 49); 
        read_count++; 
    }
    for(i=0; i<read_count; ++i) { 
        events_to_read[i][PAPI_MAX_STR_LEN - 1] = '\0'; 
        event_short_names[i][49] = '\0'; 
    }


    if (read_count == 0) { 
        fprintf(stderr, "    Error: No valid event names provided for reading.\n"); 
        return; 
    }

    long long* values = (long long*)calloc(read_count, sizeof(long long));
    if (!values) { 
        fprintf(stderr, "    Error: Failed to allocate memory for reading values.\n"); 
        return; 
    }

    CALL_PAPI_OK(PAPI_create_eventset(&readSet));
    CALL_PAPI_OK(PAPI_assign_eventset_component(readSet, cid));

    int added_count = 0;
    for (i = 0; i < read_count; ++i) {
        ret = PAPI_add_named_event(readSet, events_to_read[i]);
        if (ret == PAPI_OK) { 
            added_count++; 
            added_flags[i] = true; 
        }
        else { 
            fprintf(stderr, "    Warning: Failed to add event '%s' for reading: %s\n", 
                    events_to_read[i], PAPI_strerror(ret)); 
            if(ret == PAPI_ENOMEM){ 
                break; 
            } 
        }
    }

    if (added_count > 0) {
        CALL_PAPI_OK(PAPI_start(readSet));
        ret = PAPI_stop(readSet, values);
        if (ret != PAPI_OK){ fprintf(stderr, "    Error: PAPI_stop failed during read: %s\n", PAPI_strerror(ret)); printf("    Current System Values (PAPI_stop failed, results may be inaccurate):\n"); }
        else { printf("    Current System Values:\n"); }

        int value_idx = 0;
        for (i = 0; i < read_count; ++i) {
            if(added_flags[i]) {
                printf("      %-20s (%s): %lld\n", event_short_names[i], events_to_read[i], (ret == PAPI_OK) ? values[value_idx] : -999);
                value_idx++;
            } else { printf("      %-20s (%s): [Read Skipped - Add Failed]\n", event_short_names[i], events_to_read[i]); }
        }
    } else { printf("    No events were successfully added to the EventSet for reading.\n"); }

    free(values);
    CALL_PAPI_OK(PAPI_cleanup_eventset(readSet));
    CALL_PAPI_OK(PAPI_destroy_eventset(&readSet));
    printf("    Finished Reading [%s] --\n", stage_label);
}
