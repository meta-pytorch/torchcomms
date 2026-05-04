#include <string.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <rocm_smi.h>
#include <inttypes.h>
#include "papi.h"
#include "papi_memory.h"
#include "rocs.h"
#include "htable.h"

unsigned int _rocm_smi_lock;

static rsmi_status_t (*rsmi_num_monitor_dev_p)(uint32_t *);
static rsmi_status_t (*rsmi_func_iter_value_get_p)(rsmi_func_id_iter_handle_t, rsmi_func_id_value_t *);
static rsmi_status_t (*rsmi_func_iter_next_p)(rsmi_func_id_iter_handle_t);
static rsmi_status_t (*rsmi_dev_supported_func_iterator_open_p)(uint32_t, rsmi_func_id_iter_handle_t *);
static rsmi_status_t (*rsmi_dev_supported_func_iterator_close_p)(rsmi_func_id_iter_handle_t *);
static rsmi_status_t (*rsmi_dev_supported_variant_iterator_open_p)(rsmi_func_id_iter_handle_t, rsmi_func_id_iter_handle_t *);
static rsmi_status_t (*rsmi_dev_id_get_p)(uint32_t, uint16_t *);
static rsmi_status_t (*rsmi_dev_unique_id_get_p)(uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_brand_get_p)(uint32_t, char *, uint32_t);
static rsmi_status_t (*rsmi_dev_name_get_p)(uint32_t, char *, size_t);
static rsmi_status_t (*rsmi_dev_serial_number_get_p)(uint32_t, char *, uint32_t);
static rsmi_status_t (*rsmi_dev_vbios_version_get_p)(uint32_t, char *, uint32_t);
static rsmi_status_t (*rsmi_dev_vendor_name_get_p)(uint32_t, char *, size_t);
static rsmi_status_t (*rsmi_dev_vendor_id_get_p)(uint32_t, uint16_t *);
static rsmi_status_t (*rsmi_dev_subsystem_id_get_p)(uint32_t, uint16_t *);
static rsmi_status_t (*rsmi_dev_subsystem_vendor_id_get_p)(uint32_t, uint16_t *);
static rsmi_status_t (*rsmi_dev_subsystem_name_get_p)(uint32_t, char *, size_t);
static rsmi_status_t (*rsmi_dev_drm_render_minor_get_p)(uint32_t, uint32_t *);
static rsmi_status_t (*rsmi_dev_overdrive_level_get_p)(uint32_t, uint32_t *);
static rsmi_status_t (*rsmi_dev_overdrive_level_set_p)(uint32_t, uint32_t);
static rsmi_status_t (*rsmi_dev_memory_busy_percent_get_p)(uint32_t, uint32_t *);
static rsmi_status_t (*rsmi_dev_memory_reserved_pages_get_p)(uint32_t, uint32_t *, rsmi_retired_page_record_t *);
static rsmi_status_t (*rsmi_dev_memory_total_get_p)(uint32_t, rsmi_memory_type_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_memory_usage_get_p)(uint32_t, rsmi_memory_type_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_perf_level_get_p)(uint32_t, rsmi_dev_perf_level_t *);
static rsmi_status_t (*rsmi_dev_perf_level_set_p)(int32_t, rsmi_dev_perf_level_t);
static rsmi_status_t (*rsmi_dev_busy_percent_get_p)(uint32_t, uint32_t *);
static rsmi_status_t (*rsmi_dev_firmware_version_get_p)(uint32_t, rsmi_fw_block_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_ecc_count_get_p)(uint32_t, rsmi_gpu_block_t, rsmi_error_count_t *);
static rsmi_status_t (*rsmi_dev_ecc_enabled_get_p)(uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_ecc_status_get_p)(uint32_t, rsmi_gpu_block_t, rsmi_ras_err_state_t *);
static rsmi_status_t (*rsmi_dev_fan_reset_p)(uint32_t, uint32_t);
static rsmi_status_t (*rsmi_dev_fan_rpms_get_p)(uint32_t, uint32_t, int64_t *);
static rsmi_status_t (*rsmi_dev_fan_speed_get_p)(uint32_t, uint32_t, int64_t *);
static rsmi_status_t (*rsmi_dev_fan_speed_max_get_p)(uint32_t, uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_fan_speed_set_p)(uint32_t, uint32_t, uint64_t);
static rsmi_status_t (*rsmi_dev_power_avg_get_p)(uint32_t, uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_power_cap_get_p)(uint32_t, uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_power_cap_set_p)(uint32_t, uint32_t, uint64_t);
static rsmi_status_t (*rsmi_dev_power_cap_range_get_p)(uint32_t, uint32_t, uint64_t *, uint64_t *);
static rsmi_status_t (*rsmi_dev_power_profile_presets_get_p)(uint32_t, uint32_t, rsmi_power_profile_status_t *);
static rsmi_status_t (*rsmi_dev_power_profile_set_p)(uint32_t, uint32_t, rsmi_power_profile_preset_masks_t);
static rsmi_status_t (*rsmi_dev_temp_metric_get_p)(uint32_t, uint32_t, rsmi_temperature_metric_t, int64_t *);
static rsmi_status_t (*rsmi_dev_pci_id_get_p)(uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_pci_throughput_get_p)(uint32_t, uint64_t *, uint64_t *, uint64_t *);
static rsmi_status_t (*rsmi_dev_pci_replay_counter_get_p)(uint32_t, uint64_t *);
static rsmi_status_t (*rsmi_dev_pci_bandwidth_get_p)(uint32_t, rsmi_pcie_bandwidth_t *);
static rsmi_status_t (*rsmi_dev_pci_bandwidth_set_p)(uint32_t, uint64_t);
static rsmi_status_t (*rsmi_dev_gpu_clk_freq_get_p)(uint32_t, rsmi_clk_type_t, rsmi_frequencies_t *);
static rsmi_status_t (*rsmi_dev_gpu_clk_freq_set_p)(uint32_t, rsmi_clk_type_t, uint64_t);
static rsmi_status_t (*rsmi_dev_od_volt_curve_regions_get_p)(uint32_t, uint32_t *, rsmi_freq_volt_region_t *);
static rsmi_status_t (*rsmi_dev_od_volt_info_get_p)(uint32_t, rsmi_od_volt_freq_data_t *);
static rsmi_status_t (*rsmi_init_p)(uint64_t);
static rsmi_status_t (*rsmi_shut_down_p)(void);
static rsmi_status_t (*rsmi_version_get_p)(rsmi_version_t *);
static rsmi_status_t (*rsmi_version_str_get_p)(rsmi_sw_component_t, char *, size_t);
static rsmi_status_t (*rsmi_status_string_p)(rsmi_status_t, const char **);
static rsmi_status_t (*rsmi_dev_counter_group_supported_p)(uint32_t, rsmi_event_group_t);
static rsmi_status_t (*rsmi_dev_counter_create_p)(uint32_t, rsmi_event_type_t, rsmi_event_handle_t *);
static rsmi_status_t (*rsmi_dev_counter_destroy_p)(rsmi_event_handle_t);
static rsmi_status_t (*rsmi_counter_control_p)(rsmi_event_handle_t, rsmi_counter_command_t, void *);
static rsmi_status_t (*rsmi_counter_read_p)(rsmi_event_type_t, rsmi_counter_value_t *);
static rsmi_status_t (*rsmi_counter_available_counters_get_p)(uint32_t, rsmi_event_group_t, uint32_t *);
static rsmi_status_t (*rsmi_is_P2P_accessible_p)(uint32_t, uint32_t, int *);
static rsmi_status_t (*rsmi_minmax_bandwidth_get_p)(uint32_t, uint32_t, uint64_t *, uint64_t *);

/*
 * rocs defined variant and subvariant
 *
 * Given a rsmi function (e.g. rsmi_dev_pci_bandwidth_get) variants
 * and subvariant allow for accounting additional events associated
 * to it (e.g. ROCS_PCI_BW_VARIANT__COUNT)
 *
 */
typedef enum {
    ROCS_ACCESS_MODE__READ = 1,
    ROCS_ACCESS_MODE__WRITE,
    ROCS_ACCESS_MODE__RDWR,
} rocs_access_mode_e;

typedef enum {
    ROCS_PCI_THROUGHPUT_VARIANT__SENT,
    ROCS_PCI_THROUGHPUT_VARIANT__RECEIVED,
    ROCS_PCI_THROUGHPUT_VARIANT__MAX_PACKET_SIZE,
    ROCS_PCI_THROUGHPUT_VARIANT__NUM,
} rocs_pci_throughput_variant_e;

typedef enum {
    ROCS_POWER_PRESETS_VARIANT__COUNT,
    ROCS_POWER_PRESETS_VARIANT__AVAIL_PROFILES,
    ROCS_POWER_PRESETS_VARIANT__CURRENT,
    ROCS_POWER_PRESETS_VARIANT__NUM,
} rocs_power_presets_variant_e;

typedef enum {
    ROCS_POWER_CAP_RANGE_VARIANT__MIN,
    ROCS_POWER_CAP_RANGE_VARIANT__MAX,
    ROCS_POWER_CAP_RANGE_VARIANT__NUM,
} rocs_power_cap_range_variant_e;

typedef enum {
    ROCS_ECC_COUNT_SUBVARIANT__CORRECTABLE,
    ROCS_ECC_COUNT_SUBVARIANT__UNCORRECTABLE,
    ROCS_ECC_COUNT_SUBVARIANT__NUM,
} rocs_ecc_count_subvariant_e;

typedef enum {
    ROCS_GPU_CLK_FREQ_VARIANT__SYSTEM = RSMI_CLK_TYPE_SYS,
    ROCS_GPU_CLK_FREQ_VARIANT__DATA_FABRIC = RSMI_CLK_TYPE_DF,
    ROCS_GPU_CLK_FREQ_VARIANT__DISPLAY_ENGINE = RSMI_CLK_TYPE_DCEF,
    ROCS_GPU_CLK_FREQ_VARIANT__SOC = RSMI_CLK_TYPE_SOC,
    ROCS_GPU_CLK_FREQ_VARIANT__MEMORY = RSMI_CLK_TYPE_MEM,
    ROCS_GPU_CLK_FREQ_VARIANT__NUM,
} rocs_gpu_clk_freq_variant_e;

typedef enum {
    ROCS_GPU_CLK_FREQ_SUBVARIANT__COUNT,
    ROCS_GPU_CLK_FREQ_SUBVARIANT__CURRENT,
    ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM,
} rocs_gpu_clk_freq_subvariant_e;

typedef enum {
    ROCS_PCI_BW_VARIANT__COUNT,
    ROCS_PCI_BW_VARIANT__CURRENT,
    ROCS_PCI_BW_VARIANT__RATE_IDX,
    ROCS_PCI_BW_VARIANT__LANE_IDX,
    ROCS_PCI_BW_VARIANT__NUM,
} rocs_pci_bw_variant_e;

typedef enum {
    /* XXX: the following events (variants) and the corresponding logic
            are not tested */
    ROCS_XGMI_VARIANT__MI50_0_NOP_TX = RSMI_EVNT_XGMI_0_NOP_TX,
    ROCS_XGMI_VARIANT__MI50_0_REQUEST_TX = RSMI_EVNT_XGMI_0_REQUEST_TX,
    ROCS_XGMI_VARIANT__MI50_0_RESPONSE_TX = RSMI_EVNT_XGMI_0_RESPONSE_TX,
    ROCS_XGMI_VARIANT__MI50_0_BEATS_TX = RSMI_EVNT_XGMI_0_BEATS_TX,
    ROCS_XGMI_VARIANT__MI50_1_NOP_TX = RSMI_EVNT_XGMI_1_NOP_TX,
    ROCS_XGMI_VARIANT__MI50_1_REQUEST_TX = RSMI_EVNT_XGMI_1_REQUEST_TX,
    ROCS_XGMI_VARIANT__MI50_1_RESPONSE_TX = RSMI_EVNT_XGMI_1_RESPONSE_TX,
    ROCS_XGMI_VARIANT__MI50_1_BEATS_TX = RSMI_EVNT_XGMI_1_BEATS_TX,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_0 = RSMI_EVNT_XGMI_DATA_OUT_0,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_1 = RSMI_EVNT_XGMI_DATA_OUT_1,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_2 = RSMI_EVNT_XGMI_DATA_OUT_2,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_3 = RSMI_EVNT_XGMI_DATA_OUT_3,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_4 = RSMI_EVNT_XGMI_DATA_OUT_4,
    ROCS_XGMI_VARIANT__MI100_DATA_OUT_5 = RSMI_EVNT_XGMI_DATA_OUT_5,
} rocs_xgmi_variant_e;

typedef enum {
    ROCS_XGMI_BW_VARIANT__MIN,
    ROCS_XGMI_BW_VARIANT__MAX,
    ROCS_XGMI_BW_VARIANT__NUM,
} rocs_xgmi_bw_variant_e;

static int open_simple(void *);
static int close_simple(void *);
static int start_simple(void *);
static int stop_simple(void *);
static int open_xgmi_evt(void *);
static int close_xgmi_evt(void *);
static int start_xgmi_evt(void *);
static int stop_xgmi_evt(void *);
static int access_xgmi_evt(rocs_access_mode_e, void *);
static int access_xgmi_bw(rocs_access_mode_e, void *);
static int access_rsmi_dev_count(rocs_access_mode_e, void *);
static int access_rsmi_lib_version(rocs_access_mode_e, void *);
static int access_rsmi_dev_driver_version_str(rocs_access_mode_e, void *);
static int access_rsmi_dev_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_subsystem_vendor_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_vendor_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_unique_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_subsystem_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_drm_render_minor(rocs_access_mode_e, void *);
static int access_rsmi_dev_overdrive_level(rocs_access_mode_e, void *);
static int access_rsmi_dev_perf_level(rocs_access_mode_e, void *);
static int access_rsmi_dev_memory_total(rocs_access_mode_e, void *);
static int access_rsmi_dev_memory_usage(rocs_access_mode_e, void *);
static int access_rsmi_dev_memory_busy_percent(rocs_access_mode_e, void *);
static int access_rsmi_dev_busy_percent(rocs_access_mode_e, void *);
static int access_rsmi_dev_pci_id(rocs_access_mode_e, void *);
static int access_rsmi_dev_pci_replay_counter(rocs_access_mode_e, void *);
static int access_rsmi_dev_pci_throughput(rocs_access_mode_e, void *);
static int access_rsmi_dev_power_profile_presets(rocs_access_mode_e, void *);
static int access_rsmi_dev_power_profile_set(rocs_access_mode_e, void *);
static int access_rsmi_dev_fan_reset(rocs_access_mode_e, void *);
static int access_rsmi_dev_fan_rpms(rocs_access_mode_e, void *);
static int access_rsmi_dev_fan_speed_max(rocs_access_mode_e, void *);
static int access_rsmi_dev_fan_speed(rocs_access_mode_e, void *);
static int access_rsmi_dev_power_ave(rocs_access_mode_e, void *);
static int access_rsmi_dev_power_cap(rocs_access_mode_e, void *);
static int access_rsmi_dev_power_cap_range(rocs_access_mode_e, void *);
static int access_rsmi_dev_temp_metric(rocs_access_mode_e, void *);
static int access_rsmi_dev_firmware_version(rocs_access_mode_e, void *);
static int access_rsmi_dev_ecc_count(rocs_access_mode_e, void *);
static int access_rsmi_dev_ecc_enabled(rocs_access_mode_e, void *);
static int access_rsmi_dev_ecc_status(rocs_access_mode_e, void *);
static int access_rsmi_dev_gpu_clk_freq(rocs_access_mode_e, void *);
static int access_rsmi_dev_pci_bandwidth(rocs_access_mode_e, void *);
static int access_rsmi_dev_brand(rocs_access_mode_e, void *);
static int access_rsmi_dev_name(rocs_access_mode_e, void *);
static int access_rsmi_dev_serial_number(rocs_access_mode_e, void *);
static int access_rsmi_dev_subsystem_name(rocs_access_mode_e, void *);
static int access_rsmi_dev_vbios_version(rocs_access_mode_e, void *);
static int access_rsmi_dev_vendor_name(rocs_access_mode_e, void *);

typedef int (*open_function_f)(void *arg);
typedef int (*close_function_f)(void *arg);
typedef int (*start_function_f)(void *arg);
typedef int (*stop_function_f)(void *arg);
typedef int (*access_function_f)(rocs_access_mode_e mode, void *arg);

struct {
    const char *name;
    open_function_f open_func_p;
    close_function_f close_func_p;
    start_function_f start_func_p;
    stop_function_f stop_func_p;
    access_function_f access_func_p;
} event_function_table[] = {
    {"rsmi_dev_count", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_count},
    {"rsmi_lib_version", open_simple, close_simple, start_simple, stop_simple, access_rsmi_lib_version},
    {"rsmi_dev_driver_version_str_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_driver_version_str},
    {"rsmi_dev_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_id},
    {"rsmi_dev_subsystem_vendor_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_subsystem_vendor_id},
    {"rsmi_dev_vendor_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_vendor_id},
    {"rsmi_dev_unique_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_unique_id},
    {"rsmi_dev_subsystem_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_subsystem_id},
    {"rsmi_dev_drm_render_minor_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_drm_render_minor},
    {"rsmi_dev_overdrive_level_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_overdrive_level},
    {"rsmi_dev_overdrive_level_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_overdrive_level},
    {"rsmi_dev_perf_level_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_perf_level},
    {"rsmi_dev_perf_level_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_perf_level},
    {"rsmi_dev_memory_total_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_memory_total},
    {"rsmi_dev_memory_usage_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_memory_usage},
    {"rsmi_dev_memory_busy_percent_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_memory_busy_percent},
    {"rsmi_dev_busy_percent_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_busy_percent},
    {"rsmi_dev_pci_id_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_pci_id},
    {"rsmi_dev_pci_replay_counter_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_pci_replay_counter},
    {"rsmi_dev_pci_throughput_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_pci_throughput},
    {"rsmi_dev_power_profile_presets_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_profile_presets},
    {"rsmi_dev_power_profile_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_profile_set},
    {"rsmi_dev_fan_reset", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_fan_reset},
    {"rsmi_dev_fan_rpms_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_fan_rpms},
    {"rsmi_dev_fan_speed_max_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_fan_speed_max},
    {"rsmi_dev_fan_speed_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_fan_speed},
    {"rsmi_dev_fan_speed_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_fan_speed},
    {"rsmi_dev_power_ave_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_ave},
    {"rsmi_dev_power_cap_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_cap},
    {"rsmi_dev_power_cap_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_cap},
    {"rsmi_dev_power_cap_range_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_power_cap_range},
    {"rsmi_dev_temp_metric_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_temp_metric},
    {"rsmi_dev_firmware_version_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_firmware_version},
    {"rsmi_dev_ecc_count_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_ecc_count},
    {"rsmi_dev_ecc_enabled_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_ecc_enabled},
    {"rsmi_dev_ecc_status_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_ecc_status},
    {"rsmi_dev_gpu_clk_freq_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_gpu_clk_freq},
    {"rsmi_dev_gpu_clk_freq_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_gpu_clk_freq},
    {"rsmi_dev_pci_bandwidth_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_pci_bandwidth},
    {"rsmi_dev_pci_bandwidth_set", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_pci_bandwidth},
    {"rsmi_dev_brand_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_brand},
    {"rsmi_dev_name_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_name},
    {"rsmi_dev_serial_number_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_serial_number},
    {"rsmi_dev_subsystem_name_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_subsystem_name},
    {"rsmi_dev_vbios_version_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_vbios_version},
    {"rsmi_dev_vendor_name_get", open_simple, close_simple, start_simple, stop_simple, access_rsmi_dev_vendor_name},
    {"rsmi_dev_xgmi_evt_get", open_xgmi_evt, close_xgmi_evt, start_xgmi_evt, stop_xgmi_evt, access_xgmi_evt},
    {"rsmi_dev_xgmi_bw_get", open_simple, close_simple, start_simple, stop_simple, access_xgmi_bw},
    {NULL, NULL, NULL, NULL, NULL, NULL}
};

typedef struct {
    unsigned int id;
    char *name;
    char *descr;
    int32_t device;
    int64_t variant;
    int64_t subvariant;
    int64_t value;
    char scratch[PAPI_MAX_STR_LEN];
    rocs_access_mode_e mode;
    open_function_f open_func_p;
    close_function_f close_func_p;
    start_function_f start_func_p;
    stop_function_f stop_func_p;
    access_function_f access_func_p;
} ntv_event_t;

typedef struct ntv_event_table {
    ntv_event_t *events;
    int count;
} ntv_event_table_t;

struct rocs_ctx {
    int state;
    unsigned int *events_id;
    int num_events;
    int64_t *counters;
    int32_t device_mask;
};

#define PAPI_ROCMSMI_MAX_DEV_COUNT (32)
#define PAPI_ROCMSMI_MAX_SUBVAR    (32)

static int load_rsmi_sym(void);
static int unload_rsmi_sym(void);
static int init_event_table(void);
static int shutdown_event_table(void);
static int init_device_table(void);
static int shutdown_device_table(void);

static void *rsmi_dlp;
static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p;
static void *htable;
static char error_string[PAPI_MAX_STR_LEN + 1];
static int32_t device_count;
static rsmi_frequencies_t *freq_table;
static rsmi_pcie_bandwidth_t *pcie_table;

int
rocs_init(void)
{
    int papi_errno;

    papi_errno = load_rsmi_sym();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rsmi_status_t status = rsmi_init_p(0);
    if (status != RSMI_STATUS_SUCCESS) {
        const char *status_string = NULL;
        rsmi_status_string_p(status, &status_string);
        strcpy(error_string, status_string);
        return PAPI_EMISC;
    }

    htable_init(&htable);

    status = rsmi_num_monitor_dev_p((uint32_t *)&device_count);
    if (status != RSMI_STATUS_SUCCESS) {
        sprintf(error_string, "Error while counting available devices.");
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = init_device_table();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while initializing device tables.");
        goto fn_fail;
    }

    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while initializing the native event table.");
        goto fn_fail;
    }

    ntv_table_p = &ntv_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    shutdown_event_table();
    shutdown_device_table();
    htable_shutdown(htable);
    rsmi_shut_down_p();
    goto fn_exit;
}

int
rocs_evt_enum(unsigned int *event_code, int modifier)
{
    int papi_errno = PAPI_OK;

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if (ntv_table_p->count == 0) {
                papi_errno = PAPI_ENOEVNT;
            }
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code + 1 < (unsigned int) ntv_table_p->count) {
                ++(*event_code);
            } else {
                papi_errno = PAPI_END;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int
rocs_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    /* FIXME: make sure descr is not longer than len */
    strncpy(descr, ntv_table_p->events[event_code].descr, len);
    return PAPI_OK;
}

int
rocs_evt_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = PAPI_OK;
    int htable_errno;

    ntv_event_t *event;
    htable_errno = htable_find(htable, name, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ?
            PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }
    *event_code = event->id;

  fn_exit:
    return papi_errno;
}

int
rocs_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    strncpy(name, ntv_table_p->events[event_code].name, len);
    return PAPI_OK;
}

int
rocs_err_get_last(const char **err_string)
{
    *err_string = error_string;
    return PAPI_OK;
}

static int32_t device_mask;
static int acquire_devices(unsigned int *, int, int32_t *);
static int release_devices(int32_t *);

int
rocs_ctx_open(unsigned int *events_id, int num_events, rocs_ctx_t *rocs_ctx)
{
    int papi_errno = PAPI_OK;
    int64_t *counters = NULL;
    int i = 0, j;

    _papi_hwi_lock(_rocm_smi_lock);

    int32_t bitmask;
    if (acquire_devices(events_id, num_events, &bitmask) != PAPI_OK) {
        papi_errno = PAPI_ECNFLCT;
        goto fn_fail;
    }

    (*rocs_ctx) = papi_calloc(1, sizeof(struct rocs_ctx));
    if ((*rocs_ctx) == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    counters = papi_calloc(num_events, sizeof(int64_t));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    for (i = 0; i < num_events; ++i) {
        int id = events_id[i];
        papi_errno = ntv_table_p->events[id].open_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

    (*rocs_ctx)->state |= ROCS_EVENTS_OPENED;
    (*rocs_ctx)->events_id = events_id;
    (*rocs_ctx)->num_events = num_events;
    (*rocs_ctx)->counters = counters;
    (*rocs_ctx)->device_mask = bitmask;

  fn_exit:
    _papi_hwi_unlock(_rocm_smi_lock);
    return papi_errno;
  fn_fail:
    for (j = 0; j < i; ++j) {
        int id = events_id[j];
        ntv_table_p->events[id].close_func_p(&ntv_table_p->events[id]);
    }
    if (counters) {
        papi_free(counters);
    }
    if (*rocs_ctx) {
        papi_free(*rocs_ctx);
    }
    goto fn_exit;
}

int
rocs_ctx_close(rocs_ctx_t rocs_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_smi_lock);

    int i;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].close_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

    release_devices(&rocs_ctx->device_mask);
    papi_free(rocs_ctx->counters);
    papi_free(rocs_ctx);

  fn_exit:
    _papi_hwi_unlock(_rocm_smi_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocs_ctx_start(rocs_ctx_t rocs_ctx)
{
    int papi_errno = PAPI_OK;

    if (!(rocs_ctx->state & ROCS_EVENTS_OPENED)) {
        return PAPI_ECMP;
    }

    if (rocs_ctx->state & ROCS_EVENTS_RUNNING) {
        return PAPI_ECMP;
    }

    int i, j;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].start_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

    rocs_ctx->state |= ROCS_EVENTS_RUNNING;

  fn_exit:
    return papi_errno;
  fn_fail:
    for (j = 0; j < i; ++j) {
        int id = rocs_ctx->events_id[i];
        ntv_table_p->events[id].stop_func_p(&ntv_table_p->events[id]);
    }
    goto fn_exit;
}

int
rocs_ctx_stop(rocs_ctx_t rocs_ctx)
{
    if (!(rocs_ctx->state & ROCS_EVENTS_OPENED)) {
        return PAPI_ECMP;
    }

    if (!(rocs_ctx->state & ROCS_EVENTS_RUNNING)) {
        return PAPI_ECMP;
    }

    int i;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        int papi_errno = ntv_table_p->events[id].stop_func_p(&ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    rocs_ctx->state &= ~ROCS_EVENTS_RUNNING;

    return PAPI_OK;
}

int
rocs_ctx_read(rocs_ctx_t rocs_ctx, long long **counts)
{
    int papi_errno = PAPI_OK;

    int i;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        papi_errno = ntv_table_p->events[id].access_func_p(ROCS_ACCESS_MODE__READ, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        rocs_ctx->counters[i] = ntv_table_p->events[id].value;
    }
    *counts = (long long *) rocs_ctx->counters;

    return papi_errno;
}

int
rocs_ctx_write(rocs_ctx_t rocs_ctx, long long *counts)
{
    int papi_errno = PAPI_OK;

    int i;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        ntv_table_p->events[id].value = counts[i];
        papi_errno = ntv_table_p->events[id].access_func_p(ROCS_ACCESS_MODE__WRITE, &ntv_table_p->events[id]);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    return papi_errno;
}

int
rocs_ctx_reset(rocs_ctx_t rocs_ctx)
{
    int i;
    for (i = 0; i < rocs_ctx->num_events; ++i) {
        int id = rocs_ctx->events_id[i];
        ntv_table_p->events[id].value = 0;
        rocs_ctx->counters[i] = 0;
    }

    return PAPI_OK;
}

int
rocs_shutdown(void)
{
    shutdown_device_table();
    shutdown_event_table();
    htable_shutdown(htable);
    rsmi_shut_down_p();
    return unload_rsmi_sym();
}

int
load_rsmi_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = { 0 };
    char *rocmsmi_root = getenv("PAPI_ROCMSMI_ROOT");
    if (rocmsmi_root == NULL) {
        sprintf(error_string, "Can't load librocm_smi64.so, PAPI_ROCMSMI_ROOT not set.");
        goto fn_fail;
    }

    sprintf(pathname, "%s/lib/librocm_smi64.so", rocmsmi_root);

    rsmi_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rsmi_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    rsmi_num_monitor_dev_p                     = dlsym(rsmi_dlp, "rsmi_num_monitor_devices");
    rsmi_func_iter_value_get_p                 = dlsym(rsmi_dlp, "rsmi_func_iter_value_get");
    rsmi_func_iter_next_p                      = dlsym(rsmi_dlp, "rsmi_func_iter_next");
    rsmi_dev_supported_func_iterator_open_p    = dlsym(rsmi_dlp, "rsmi_dev_supported_func_iterator_open");
    rsmi_dev_supported_func_iterator_close_p   = dlsym(rsmi_dlp, "rsmi_dev_supported_func_iterator_close");
    rsmi_dev_supported_variant_iterator_open_p = dlsym(rsmi_dlp, "rsmi_dev_supported_variant_iterator_open");
    rsmi_dev_id_get_p                          = dlsym(rsmi_dlp, "rsmi_dev_id_get");
    rsmi_dev_unique_id_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_unique_id_get");
    rsmi_dev_brand_get_p                       = dlsym(rsmi_dlp, "rsmi_dev_brand_get");
    rsmi_dev_name_get_p                        = dlsym(rsmi_dlp, "rsmi_dev_name_get");
    rsmi_dev_serial_number_get_p               = dlsym(rsmi_dlp, "rsmi_dev_serial_number_get");
    rsmi_dev_vbios_version_get_p               = dlsym(rsmi_dlp, "rsmi_dev_vbios_version_get");
    rsmi_dev_vendor_name_get_p                 = dlsym(rsmi_dlp, "rsmi_dev_vendor_name_get");
    rsmi_dev_vendor_id_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_vendor_id_get");
    rsmi_dev_subsystem_id_get_p                = dlsym(rsmi_dlp, "rsmi_dev_subsystem_id_get");
    rsmi_dev_subsystem_vendor_id_get_p         = dlsym(rsmi_dlp, "rsmi_dev_subsystem_vendor_id_get");
    rsmi_dev_subsystem_name_get_p              = dlsym(rsmi_dlp, "rsmi_dev_subsystem_name_get");
    rsmi_dev_drm_render_minor_get_p            = dlsym(rsmi_dlp, "rsmi_dev_drm_render_minor_get");
    rsmi_dev_overdrive_level_get_p             = dlsym(rsmi_dlp, "rsmi_dev_overdrive_level_get");
    rsmi_dev_overdrive_level_set_p             = dlsym(rsmi_dlp, "rsmi_dev_overdrive_level_set");
    rsmi_dev_memory_busy_percent_get_p         = dlsym(rsmi_dlp, "rsmi_dev_memory_busy_percent_get");
    rsmi_dev_memory_reserved_pages_get_p       = dlsym(rsmi_dlp, "rsmi_dev_memory_reserved_pages_get");
    rsmi_dev_memory_total_get_p                = dlsym(rsmi_dlp, "rsmi_dev_memory_total_get");
    rsmi_dev_memory_usage_get_p                = dlsym(rsmi_dlp, "rsmi_dev_memory_usage_get");
    rsmi_dev_perf_level_get_p                  = dlsym(rsmi_dlp, "rsmi_dev_perf_level_get");
    rsmi_dev_perf_level_set_p                  = dlsym(rsmi_dlp, "rsmi_dev_perf_level_set");
    rsmi_dev_busy_percent_get_p                = dlsym(rsmi_dlp, "rsmi_dev_busy_percent_get");
    rsmi_dev_firmware_version_get_p            = dlsym(rsmi_dlp, "rsmi_dev_firmware_version_get");
    rsmi_dev_ecc_count_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_ecc_count_get");
    rsmi_dev_ecc_enabled_get_p                 = dlsym(rsmi_dlp, "rsmi_dev_ecc_enabled_get");
    rsmi_dev_ecc_status_get_p                  = dlsym(rsmi_dlp, "rsmi_dev_ecc_status_get");
    rsmi_dev_fan_reset_p                       = dlsym(rsmi_dlp, "rsmi_dev_fan_reset");
    rsmi_dev_fan_rpms_get_p                    = dlsym(rsmi_dlp, "rsmi_dev_fan_rpms_get");
    rsmi_dev_fan_speed_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_fan_speed_get");
    rsmi_dev_fan_speed_max_get_p               = dlsym(rsmi_dlp, "rsmi_dev_fan_speed_max_get");
    rsmi_dev_fan_speed_set_p                   = dlsym(rsmi_dlp, "rsmi_dev_fan_speed_set");
    rsmi_dev_power_avg_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_power_ave_get");
    rsmi_dev_power_cap_get_p                   = dlsym(rsmi_dlp, "rsmi_dev_power_cap_get");
    rsmi_dev_power_cap_set_p                   = dlsym(rsmi_dlp, "rsmi_dev_power_cap_set");
    rsmi_dev_power_cap_range_get_p             = dlsym(rsmi_dlp, "rsmi_dev_power_cap_range_get");
    rsmi_dev_power_profile_presets_get_p       = dlsym(rsmi_dlp, "rsmi_dev_power_profile_presets_get");
    rsmi_dev_power_profile_set_p               = dlsym(rsmi_dlp, "rsmi_dev_power_profile_set");
    rsmi_dev_temp_metric_get_p                 = dlsym(rsmi_dlp, "rsmi_dev_temp_metric_get");
    rsmi_dev_pci_id_get_p                      = dlsym(rsmi_dlp, "rsmi_dev_pci_id_get");
    rsmi_dev_pci_throughput_get_p              = dlsym(rsmi_dlp, "rsmi_dev_pci_throughput_get");
    rsmi_dev_pci_replay_counter_get_p          = dlsym(rsmi_dlp, "rsmi_dev_pci_replay_counter_get");
    rsmi_dev_pci_bandwidth_get_p               = dlsym(rsmi_dlp, "rsmi_dev_pci_bandwidth_get");
    rsmi_dev_pci_bandwidth_set_p               = dlsym(rsmi_dlp, "rsmi_dev_pci_bandwidth_set");
    rsmi_dev_gpu_clk_freq_get_p                = dlsym(rsmi_dlp, "rsmi_dev_gpu_clk_freq_get");
    rsmi_dev_gpu_clk_freq_set_p                = dlsym(rsmi_dlp, "rsmi_dev_gpu_clk_freq_set");
    rsmi_dev_od_volt_curve_regions_get_p       = dlsym(rsmi_dlp, "rsmi_dev_od_volt_curve_regions_get");
    rsmi_dev_od_volt_info_get_p                = dlsym(rsmi_dlp, "rsmi_dev_od_volt_info_get");
    rsmi_init_p                                = dlsym(rsmi_dlp, "rsmi_init");
    rsmi_shut_down_p                           = dlsym(rsmi_dlp, "rsmi_shut_down");
    rsmi_version_get_p                         = dlsym(rsmi_dlp, "rsmi_version_get");
    rsmi_version_str_get_p                     = dlsym(rsmi_dlp, "rsmi_version_str_get");
    rsmi_status_string_p                       = dlsym(rsmi_dlp, "rsmi_status_string");
    rsmi_dev_counter_group_supported_p         = dlsym(rsmi_dlp, "rsmi_dev_counter_group_supported");
    rsmi_dev_counter_create_p                  = dlsym(rsmi_dlp, "rsmi_dev_counter_create");
    rsmi_dev_counter_destroy_p                 = dlsym(rsmi_dlp, "rsmi_dev_counter_destroy");
    rsmi_counter_control_p                     = dlsym(rsmi_dlp, "rsmi_counter_control");
    rsmi_counter_read_p                        = dlsym(rsmi_dlp, "rsmi_counter_read");
    rsmi_counter_available_counters_get_p      = dlsym(rsmi_dlp, "rsmi_counter_available_counters_get");
    rsmi_is_P2P_accessible_p                   = dlsym(rsmi_dlp, "rsmi_is_P2P_accessible");
    rsmi_minmax_bandwidth_get_p                = dlsym(rsmi_dlp, "rsmi_minmax_bandwidth_get");

    int rsmi_not_initialized = (!rsmi_num_monitor_dev_p                     ||
                                !rsmi_func_iter_value_get_p                 ||
                                !rsmi_func_iter_next_p                      ||
                                !rsmi_dev_supported_func_iterator_open_p    ||
                                !rsmi_dev_supported_func_iterator_close_p   ||
                                !rsmi_dev_supported_variant_iterator_open_p ||
                                !rsmi_dev_id_get_p                          ||
                                !rsmi_dev_unique_id_get_p                   ||
                                !rsmi_dev_brand_get_p                       ||
                                !rsmi_dev_name_get_p                        ||
                                !rsmi_dev_serial_number_get_p               ||
                                !rsmi_dev_vbios_version_get_p               ||
                                !rsmi_dev_vendor_name_get_p                 ||
                                !rsmi_dev_vendor_id_get_p                   ||
                                !rsmi_dev_subsystem_id_get_p                ||
                                !rsmi_dev_subsystem_vendor_id_get_p         ||
                                !rsmi_dev_subsystem_name_get_p              ||
                                !rsmi_dev_drm_render_minor_get_p            ||
                                !rsmi_dev_overdrive_level_get_p             ||
                                !rsmi_dev_overdrive_level_set_p             ||
                                !rsmi_dev_memory_busy_percent_get_p         ||
                                !rsmi_dev_memory_reserved_pages_get_p       ||
                                !rsmi_dev_memory_total_get_p                ||
                                !rsmi_dev_memory_usage_get_p                ||
                                !rsmi_dev_perf_level_get_p                  ||
                                !rsmi_dev_perf_level_set_p                  ||
                                !rsmi_dev_busy_percent_get_p                ||
                                !rsmi_dev_firmware_version_get_p            ||
                                !rsmi_dev_ecc_count_get_p                   ||
                                !rsmi_dev_ecc_enabled_get_p                 ||
                                !rsmi_dev_ecc_status_get_p                  ||
                                !rsmi_dev_fan_reset_p                       ||
                                !rsmi_dev_fan_rpms_get_p                    ||
                                !rsmi_dev_fan_speed_get_p                   ||
                                !rsmi_dev_fan_speed_max_get_p               ||
                                !rsmi_dev_fan_speed_set_p                   ||
                                !rsmi_dev_power_avg_get_p                   ||
                                !rsmi_dev_power_cap_get_p                   ||
                                !rsmi_dev_power_cap_set_p                   ||
                                !rsmi_dev_power_cap_range_get_p             ||
                                !rsmi_dev_power_profile_presets_get_p       ||
                                !rsmi_dev_power_profile_set_p               ||
                                !rsmi_dev_temp_metric_get_p                 ||
                                !rsmi_dev_pci_id_get_p                      ||
                                !rsmi_dev_pci_throughput_get_p              ||
                                !rsmi_dev_pci_replay_counter_get_p          ||
                                !rsmi_dev_pci_bandwidth_get_p               ||
                                !rsmi_dev_pci_bandwidth_set_p               ||
                                !rsmi_dev_gpu_clk_freq_get_p                ||
                                !rsmi_dev_gpu_clk_freq_set_p                ||
                                !rsmi_dev_od_volt_curve_regions_get_p       ||
                                !rsmi_dev_od_volt_info_get_p                ||
                                !rsmi_init_p                                ||
                                !rsmi_shut_down_p                           ||
                                !rsmi_version_get_p                         ||
                                !rsmi_version_str_get_p                     ||
                                !rsmi_status_string_p                       ||
                                !rsmi_dev_counter_group_supported_p         ||
                                !rsmi_dev_counter_create_p                  ||
                                !rsmi_dev_counter_destroy_p                 ||
                                !rsmi_counter_control_p                     ||
                                !rsmi_counter_read_p                        ||
                                !rsmi_counter_available_counters_get_p      ||
                                !rsmi_is_P2P_accessible_p                   ||
                                !rsmi_minmax_bandwidth_get_p);

    papi_errno = (rsmi_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading rocm_smi symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

int
unload_rsmi_sym(void)
{
    rsmi_num_monitor_dev_p                     = NULL;
    rsmi_func_iter_value_get_p                 = NULL;
    rsmi_func_iter_next_p                      = NULL;
    rsmi_dev_supported_func_iterator_open_p    = NULL;
    rsmi_dev_supported_func_iterator_close_p   = NULL;
    rsmi_dev_supported_variant_iterator_open_p = NULL;
    rsmi_dev_id_get_p                          = NULL;
    rsmi_dev_unique_id_get_p                   = NULL;
    rsmi_dev_brand_get_p                       = NULL;
    rsmi_dev_name_get_p                        = NULL;
    rsmi_dev_serial_number_get_p               = NULL;
    rsmi_dev_vbios_version_get_p               = NULL;
    rsmi_dev_vendor_name_get_p                 = NULL;
    rsmi_dev_vendor_id_get_p                   = NULL;
    rsmi_dev_subsystem_id_get_p                = NULL;
    rsmi_dev_subsystem_vendor_id_get_p         = NULL;
    rsmi_dev_subsystem_name_get_p              = NULL;
    rsmi_dev_drm_render_minor_get_p            = NULL;
    rsmi_dev_overdrive_level_get_p             = NULL;
    rsmi_dev_overdrive_level_set_p             = NULL;
    rsmi_dev_memory_busy_percent_get_p         = NULL;
    rsmi_dev_memory_reserved_pages_get_p       = NULL;
    rsmi_dev_memory_total_get_p                = NULL;
    rsmi_dev_memory_usage_get_p                = NULL;
    rsmi_dev_perf_level_get_p                  = NULL;
    rsmi_dev_perf_level_set_p                  = NULL;
    rsmi_dev_busy_percent_get_p                = NULL;
    rsmi_dev_firmware_version_get_p            = NULL;
    rsmi_dev_ecc_count_get_p                   = NULL;
    rsmi_dev_ecc_enabled_get_p                 = NULL;
    rsmi_dev_ecc_status_get_p                  = NULL;
    rsmi_dev_fan_reset_p                       = NULL;
    rsmi_dev_fan_rpms_get_p                    = NULL;
    rsmi_dev_fan_speed_get_p                   = NULL;
    rsmi_dev_fan_speed_max_get_p               = NULL;
    rsmi_dev_fan_speed_set_p                   = NULL;
    rsmi_dev_power_avg_get_p                   = NULL;
    rsmi_dev_power_cap_get_p                   = NULL;
    rsmi_dev_power_cap_set_p                   = NULL;
    rsmi_dev_power_cap_range_get_p             = NULL;
    rsmi_dev_power_profile_presets_get_p       = NULL;
    rsmi_dev_power_profile_set_p               = NULL;
    rsmi_dev_temp_metric_get_p                 = NULL;
    rsmi_dev_pci_id_get_p                      = NULL;
    rsmi_dev_pci_throughput_get_p              = NULL;
    rsmi_dev_pci_replay_counter_get_p          = NULL;
    rsmi_dev_pci_bandwidth_get_p               = NULL;
    rsmi_dev_pci_bandwidth_set_p               = NULL;
    rsmi_dev_gpu_clk_freq_get_p                = NULL;
    rsmi_dev_gpu_clk_freq_set_p                = NULL;
    rsmi_dev_od_volt_curve_regions_get_p       = NULL;
    rsmi_dev_od_volt_info_get_p                = NULL;
    rsmi_init_p                                = NULL;
    rsmi_shut_down_p                           = NULL;
    rsmi_version_get_p                         = NULL;
    rsmi_version_str_get_p                     = NULL;
    rsmi_status_string_p                       = NULL;
    rsmi_dev_counter_group_supported_p         = NULL;
    rsmi_dev_counter_create_p                  = NULL;
    rsmi_dev_counter_destroy_p                 = NULL;
    rsmi_counter_control_p                     = NULL;
    rsmi_counter_read_p                        = NULL;
    rsmi_counter_available_counters_get_p      = NULL;
    rsmi_is_P2P_accessible_p                   = NULL;
    rsmi_minmax_bandwidth_get_p                = NULL;

    dlclose(rsmi_dlp);

    return PAPI_OK;
}

static int get_ntv_events_count(int *count);
static int get_ntv_events(ntv_event_t *, int);

int
init_event_table(void)
{
    int papi_errno = PAPI_OK;

    int ntv_events_count;
    papi_errno = get_ntv_events_count(&ntv_events_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    ntv_event_t *ntv_events = papi_calloc(ntv_events_count, sizeof(*ntv_events));
    if (ntv_events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = get_ntv_events(ntv_events, ntv_events_count);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_table.count = ntv_events_count;
    ntv_table.events = ntv_events;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (ntv_events) {
        papi_free(ntv_events);
    }
    goto fn_exit;
}

int
shutdown_event_table(void)
{
    int i;
    for (i = 0; i < ntv_table.count; ++i) {
        htable_delete(htable, ntv_table.events[i].name);
        papi_free(ntv_table.events[i].name);
        papi_free(ntv_table.events[i].descr);
    }
    papi_free(ntv_table.events);
    ntv_table.events = NULL;
    ntv_table.count = 0;
    return PAPI_OK;
}

int
acquire_devices(unsigned int *events_id, int num_events, int32_t *bitmask)
{
    int i;
    int32_t device_mask_acq = 0;

    for (i = 0; i < num_events; ++i) {
        int32_t device_id = ntv_table_p->events[events_id[i]].device;
        if (device_id < 0) {
            continue;
        }
        device_mask_acq |= (1 << device_id);
    }

    if (device_mask_acq & device_mask) {
        return PAPI_ECNFLCT;
    }

    device_mask |= device_mask_acq;
    *bitmask = device_mask_acq;

    return PAPI_OK;
}

int
release_devices(int32_t *bitmask)
{
    int32_t device_mask_rel = *bitmask;

    if ((device_mask_rel & device_mask) != device_mask_rel) {
        return PAPI_EMISC;
    }

    device_mask ^= device_mask_rel;
    *bitmask ^= device_mask_rel;

    return PAPI_OK;
}

int
init_device_table(void)
{
    int papi_errno = PAPI_OK;
    int i, j;
    rsmi_status_t status;

    freq_table = calloc(device_count * ROCS_GPU_CLK_FREQ_VARIANT__NUM, sizeof(rsmi_frequencies_t));
    if (freq_table == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    pcie_table = calloc(device_count, sizeof(rsmi_pcie_bandwidth_t));
    if (pcie_table == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    for (i = 0; i < device_count; ++i) {
        for (j = 0; j < ROCS_GPU_CLK_FREQ_VARIANT__NUM; ++j) {
            int table_id = ROCS_GPU_CLK_FREQ_VARIANT__NUM * i + j;
            status = rsmi_dev_gpu_clk_freq_get_p(i, j, &freq_table[table_id]);
            if (status != RSMI_STATUS_SUCCESS && status != RSMI_STATUS_NOT_SUPPORTED) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }
        }
    }

    for (i = 0; i < device_count; ++i) {
        status = rsmi_dev_pci_bandwidth_get_p(i, &pcie_table[i]);
        /* 
          Retrieve available PCIe bandwidths. This function is not supported on newer hardware (i.e., MI250 and MI300), but
           supported on some hardware. Ignore statuses indicating lack of support or 
           unimplemented functionality.
        */
        if (status != RSMI_STATUS_SUCCESS && status != RSMI_STATUS_NOT_YET_IMPLEMENTED && status != RSMI_STATUS_NOT_SUPPORTED) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    if (freq_table) {
        papi_free(freq_table);
        freq_table = NULL;
    }
    if (pcie_table) {
        papi_free(pcie_table);
        pcie_table = NULL;
    }
    goto fn_exit;
}

int
shutdown_device_table(void)
{
    if (freq_table) {
        papi_free(freq_table);
    }

    if (pcie_table) {
        papi_free(pcie_table);
    }

    return PAPI_OK;
}

#define ROCMSMI_NUM_INFO_EVENTS (3)

typedef enum {
    ROCS_EVENT_TYPE__NATIVE = 0,
    ROCS_EVENT_TYPE__DERIVED,
} rocs_event_type_e;

static int handle_derived_events_count(const char *v_name, int32_t dev, int64_t v_variant, int64_t v_subvariant, int *count);
static int handle_xgmi_events_count(int32_t dev, int *count);
static char *get_event_name(const char *name, int32_t dev, int64_t variant, int64_t subvariant);

int
get_ntv_events_count(int *count)
{
    int papi_errno = PAPI_OK;
    int events_count = ROCMSMI_NUM_INFO_EVENTS;
    rsmi_func_id_iter_handle_t iter;
    rsmi_func_id_iter_handle_t var_iter;
    rsmi_func_id_iter_handle_t subvar_iter;
    rsmi_func_id_value_t v_name;
    rsmi_func_id_value_t v_variant;
    rsmi_func_id_value_t v_subvariant;
    rsmi_status_t status;

    int32_t dev;
    for (dev = 0; dev < device_count; ++dev) {
        status = rsmi_dev_supported_func_iterator_open_p(dev, &iter);
        if (status != RSMI_STATUS_SUCCESS) {
            continue;
        }
        while (1) {
            status = rsmi_func_iter_value_get_p(iter, &v_name);
            if (status != RSMI_STATUS_SUCCESS) {
                continue;
            }
            status = rsmi_dev_supported_variant_iterator_open_p(iter, &var_iter);
            if (status == RSMI_STATUS_NO_DATA) {
                if (handle_derived_events_count(v_name.name, dev, -1, -1, &events_count) == ROCS_EVENT_TYPE__NATIVE) {
                    char *name = get_event_name(v_name.name, dev, -1, -1);
                    if (name) {
                        /* count known events */
                        ++events_count;
                        papi_free(name);
                    }
                }
            } else {
                while (status != RSMI_STATUS_NO_DATA) {
                    status = rsmi_func_iter_value_get_p(var_iter, &v_variant);
                    if (status != RSMI_STATUS_SUCCESS) {
                        continue;
                    }
                    status = rsmi_dev_supported_variant_iterator_open_p(var_iter, &subvar_iter);
                    if (status == RSMI_STATUS_NO_DATA) {
                        if (handle_derived_events_count(v_name.name, dev, v_variant.id, -1, &events_count) == ROCS_EVENT_TYPE__NATIVE) {
                            char *name = get_event_name(v_name.name, dev, v_variant.id, -1);
                            if (name) {
                                /* count known events */
                                ++events_count;
                                papi_free(name);
                            }
                        }
                    } else {
                        while (status != RSMI_STATUS_NO_DATA) {
                            status = rsmi_func_iter_value_get_p(subvar_iter, &v_subvariant);
                            if (status != RSMI_STATUS_SUCCESS) {
                                continue;
                            }
                            if (handle_derived_events_count(v_name.name, dev, v_variant.id, v_subvariant.id, &events_count) == ROCS_EVENT_TYPE__NATIVE) {
                                char *name = get_event_name(v_name.name, dev, v_variant.id, v_subvariant.id);
                                if (name) {
                                    /* count known events */
                                    ++events_count;
                                    papi_free(name);
                                }
                            }
                            status = rsmi_func_iter_next_p(subvar_iter);
                        }
                        status = rsmi_dev_supported_func_iterator_close_p(&subvar_iter);
                        if (status != RSMI_STATUS_SUCCESS) {
                            papi_errno = PAPI_EMISC;
                            goto fn_fail;
                        }
                    }
                    status = rsmi_func_iter_next_p(var_iter);
                }
                status = rsmi_dev_supported_func_iterator_close_p(&var_iter);
                if (status != RSMI_STATUS_SUCCESS) {
                    papi_errno = PAPI_EMISC;
                    goto fn_fail;
                }
            }
            status = rsmi_func_iter_next_p(iter);
            if (status == RSMI_STATUS_NO_DATA) {
                break;
            }
        }
        status = rsmi_dev_supported_func_iterator_close_p(&iter);
        if (status != RSMI_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        handle_xgmi_events_count(dev, &events_count);
    }

    *count = events_count;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

static rocs_access_mode_e get_access_mode(const char *);
static char *get_event_descr(const char *name, int64_t variant, int64_t subvariant);
static open_function_f get_open_func(const char *name);
static close_function_f get_close_func(const char *name);
static start_function_f get_start_func(const char *name);
static stop_function_f get_stop_func(const char *name);
static access_function_f get_access_func(const char *name);
static int handle_derived_events(const char *name, int32_t dev, int64_t variant, int64_t subvariant, int *count, ntv_event_t *events);
static int handle_xgmi_events(int32_t dev, int *count, ntv_event_t *events);

int
get_ntv_events(ntv_event_t *events, int count)
{
    int papi_errno = PAPI_OK;
    rsmi_func_id_iter_handle_t iter;
    rsmi_func_id_iter_handle_t var_iter;
    rsmi_func_id_iter_handle_t subvar_iter;
    rsmi_func_id_value_t v_name;
    rsmi_func_id_value_t v_variant;
    rsmi_func_id_value_t v_subvariant;
    rsmi_status_t status;
    int events_count = 0;

    const char *first_events[] = {
        "rsmi_dev_count",
        "rsmi_lib_version",
        "rsmi_dev_driver_version_str_get",
    };

    int i;
    for (i = 0; i < ROCMSMI_NUM_INFO_EVENTS; ++i) {
        events[events_count].id = events_count;
        events[events_count].name = get_event_name(first_events[i], -1, -1, -1);
        events[events_count].descr = get_event_descr(first_events[i], -1, -1);
        events[events_count].device = -1;
        events[events_count].variant = -1;
        events[events_count].subvariant = -1;
        events[events_count].mode = ROCS_ACCESS_MODE__READ;
        events[events_count].open_func_p = get_open_func(first_events[i]);
        events[events_count].close_func_p = get_close_func(first_events[i]);
        events[events_count].start_func_p = get_start_func(first_events[i]);
        events[events_count].stop_func_p = get_stop_func(first_events[i]);
        events[events_count].access_func_p = get_access_func(first_events[i]);
        htable_insert(htable, events[events_count].name, &events[events_count]);
        ++events_count;
    }

    int32_t dev;
    for (dev = 0; dev < device_count; ++dev) {
        status = rsmi_dev_supported_func_iterator_open_p(dev, &iter);
        if (status != RSMI_STATUS_SUCCESS) {
            continue;
        }
        while (1) {
            status = rsmi_func_iter_value_get_p(iter, &v_name);
            if (status != RSMI_STATUS_SUCCESS) {
                continue;
            }
            status = rsmi_dev_supported_variant_iterator_open_p(iter, &var_iter);
            if (status == RSMI_STATUS_NO_DATA) {
                if (handle_derived_events(v_name.name, dev, -1, -1, &events_count, events) == ROCS_EVENT_TYPE__NATIVE) {
                    char *name = get_event_name(v_name.name, dev, -1, -1);
                    if (name) {
                        /* add known events */
                        events[events_count].id = events_count;
                        events[events_count].name = name;
                        events[events_count].descr = get_event_descr(v_name.name, -1, -1);
                        events[events_count].device = dev;
                        events[events_count].variant = -1;
                        events[events_count].subvariant = -1;
                        events[events_count].mode = get_access_mode(v_name.name);
                        events[events_count].open_func_p = get_open_func(v_name.name);
                        events[events_count].close_func_p = get_close_func(v_name.name);
                        events[events_count].start_func_p = get_start_func(v_name.name);
                        events[events_count].stop_func_p = get_stop_func(v_name.name);
                        events[events_count].access_func_p = get_access_func(v_name.name);
                        htable_insert(htable, events[events_count].name, &events[events_count]);
                        ++events_count;
                    }
                }
            } else {
                while (status != RSMI_STATUS_NO_DATA) {
                    status = rsmi_func_iter_value_get_p(var_iter, &v_variant);
                    if (status != RSMI_STATUS_SUCCESS) {
                        continue;
                    }
                    status = rsmi_dev_supported_variant_iterator_open_p(var_iter, &subvar_iter);
                    if (status == RSMI_STATUS_NO_DATA) {
                        if (handle_derived_events(v_name.name, dev, v_variant.id, -1, &events_count, events) == ROCS_EVENT_TYPE__NATIVE) {
                            char *name = get_event_name(v_name.name, dev, v_variant.id, -1);
                            if (name) {
                                /* add known events */
                                events[events_count].id = events_count;
                                events[events_count].name = name;
                                events[events_count].descr = get_event_descr(v_name.name, v_variant.id, -1);
                                events[events_count].device = dev;
                                events[events_count].variant = v_variant.id;
                                events[events_count].subvariant = -1;
                                events[events_count].open_func_p = get_open_func(v_name.name);
                                events[events_count].close_func_p = get_close_func(v_name.name);
                                events[events_count].start_func_p = get_start_func(v_name.name);
                                events[events_count].stop_func_p = get_stop_func(v_name.name);
                                events[events_count].mode = get_access_mode(v_name.name);
                                events[events_count].access_func_p = get_access_func(v_name.name);
                                htable_insert(htable, events[events_count].name, &events[events_count]);
                                ++events_count;
                            }
                        }
                    } else {
                        while (status != RSMI_STATUS_NO_DATA) {
                            status = rsmi_func_iter_value_get_p(subvar_iter, &v_subvariant);
                            if (status != RSMI_STATUS_SUCCESS) {
                                continue;
                            }
                            if (handle_derived_events(v_name.name, dev, v_variant.id, v_subvariant.id, &events_count, events) == ROCS_EVENT_TYPE__NATIVE) {
                                char *name = get_event_name(v_name.name, dev, v_variant.id, v_subvariant.id);
                                if (name) {
                                    /* add known events */
                                    events[events_count].id = events_count;
                                    events[events_count].name = name;
                                    events[events_count].descr = get_event_descr(v_name.name, v_variant.id, v_subvariant.id);
                                    events[events_count].device = dev;
                                    events[events_count].variant = v_variant.id;
                                    events[events_count].subvariant = v_subvariant.id;
                                    events[events_count].mode = get_access_mode(v_name.name);
                                    events[events_count].open_func_p = get_open_func(v_name.name);
                                    events[events_count].close_func_p = get_close_func(v_name.name);
                                    events[events_count].start_func_p = get_start_func(v_name.name);
                                    events[events_count].stop_func_p = get_stop_func(v_name.name);
                                    events[events_count].access_func_p = get_access_func(v_name.name);
                                    htable_insert(htable, events[events_count].name, &events[events_count]);
                                    ++events_count;
                                }
                            }
                            status = rsmi_func_iter_next_p(subvar_iter);
                        }
                        status = rsmi_dev_supported_func_iterator_close_p(&subvar_iter);
                        if (status != RSMI_STATUS_SUCCESS) {
                            papi_errno = PAPI_EMISC;
                            goto fn_fail;
                        }
                    }
                    status = rsmi_func_iter_next_p(var_iter);
                }
                status = rsmi_dev_supported_func_iterator_close_p(&var_iter);
                if (status != RSMI_STATUS_SUCCESS) {
                    papi_errno = PAPI_EMISC;
                    goto fn_fail;
                }
            }
            status = rsmi_func_iter_next_p(iter);
            if (status == RSMI_STATUS_NO_DATA) {
                break;
            }
        }
        status = rsmi_dev_supported_func_iterator_close_p(&iter);
        if (status != RSMI_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        handle_xgmi_events(dev, &events_count, events);
    }

    papi_errno = (events_count - count) ? PAPI_ECMP : PAPI_OK;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
handle_derived_events_count(const char *v_name, int32_t dev, int64_t v_variant, int64_t v_subvariant, int *events_count)
{
    /* NOTE: derived events are of two types:
     * (a) one rsmi event contains aggregated pieces of data and is thus
     *     split into separate events in the rocm smi component;
     * (b) two rsmi events are merged into a single one in the rocm smi
     *     component. An example is the variants set/get which are
     *     both represented by the same native event in PAPI.
     */

    if (strcmp(v_name, "rsmi_dev_pci_throughput_get") == 0) {
        (*events_count) += ROCS_PCI_THROUGHPUT_VARIANT__NUM;
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_power_profile_presets_get") == 0) {
        (*events_count) += ROCS_POWER_PRESETS_VARIANT__NUM;
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_power_cap_range_get") == 0) {
        (*events_count) += ROCS_POWER_CAP_RANGE_VARIANT__NUM;
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_ecc_count_get") == 0) {
        (*events_count) += ROCS_ECC_COUNT_SUBVARIANT__NUM;
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_pci_bandwidth_get") == 0) {
        if (pcie_table[dev].transfer_rate.num_supported) {
            (*events_count) += ROCS_PCI_BW_VARIANT__CURRENT + 1;
        }
        int i;
        for (i = 0; i < ROCS_PCI_BW_VARIANT__LANE_IDX - ROCS_PCI_BW_VARIANT__CURRENT; ++i) {
            (*events_count) += pcie_table[dev].transfer_rate.num_supported;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_pci_bandwidth_set") == 0) {
        if (pcie_table[dev].transfer_rate.num_supported) {
            ++(*events_count);
        }
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_gpu_clk_freq_get") == 0) {
        int table_id = dev * ROCS_GPU_CLK_FREQ_VARIANT__NUM + v_variant;
        (*events_count) += freq_table[table_id].num_supported;
        if (freq_table[table_id].num_supported) {
            (*events_count) += ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM;
        }
        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_gpu_clk_freq_set") == 0) {
        int table_id = dev * ROCS_GPU_CLK_FREQ_VARIANT__NUM + v_variant;
        if (freq_table[table_id].num_supported) {
            ++(*events_count);
        }
        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_fan_speed_count[PAPI_ROCMSMI_MAX_DEV_COUNT];
    if (strcmp(v_name, "rsmi_dev_fan_speed_get") == 0 || strcmp(v_name, "rsmi_dev_fan_speed_set") == 0) {
        if (rsmi_dev_fan_speed_count[dev] == 0) {
            rsmi_dev_fan_speed_count[dev] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_power_cap_count[PAPI_ROCMSMI_MAX_DEV_COUNT][PAPI_ROCMSMI_MAX_SUBVAR];
    if (strcmp(v_name, "rsmi_dev_power_cap_get") == 0 || strcmp(v_name, "rsmi_dev_power_cap_set") == 0) {
        if (rsmi_dev_power_cap_count[dev][v_subvariant] == 0) {
            rsmi_dev_power_cap_count[dev][v_subvariant] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_perf_level_count[PAPI_ROCMSMI_MAX_DEV_COUNT];
    if (strncmp(v_name, "rsmi_dev_perf_level", strlen("rsmi_dev_perf_level")) == 0) {
        if (rsmi_dev_perf_level_count[dev] == 0) {
            rsmi_dev_perf_level_count[dev] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }


    return ROCS_EVENT_TYPE__NATIVE;
}

int
handle_xgmi_events_count(int32_t dev, int *events_count)
{
    rsmi_status_t status;

    status = rsmi_dev_counter_group_supported_p(dev, RSMI_EVNT_GRP_XGMI);
    if (status == RSMI_STATUS_SUCCESS) {
        *events_count += RSMI_EVNT_XGMI_LAST - RSMI_EVNT_XGMI_FIRST;
    }

    status = rsmi_dev_counter_group_supported_p(dev, RSMI_EVNT_GRP_XGMI_DATA_OUT);
    if (status == RSMI_STATUS_SUCCESS) {
        *events_count += RSMI_EVNT_XGMI_DATA_OUT_LAST - RSMI_EVNT_XGMI_DATA_OUT_FIRST;
    }

    uint32_t i;
    for (i = 0; i < (uint32_t) device_count; ++i) {
        if (i == (uint32_t) dev) {
            continue;
        }
        rsmi_status_t status;
        int res = 0;
        status = rsmi_is_P2P_accessible_p((uint32_t) dev, i, &res);
        if (status != RSMI_STATUS_SUCCESS) {
            break;
        }
        uint64_t min, max;
        status = rsmi_minmax_bandwidth_get_p((uint32_t) dev, i, &min, &max);
        if (status != RSMI_STATUS_SUCCESS) {
            break;
        }
        if (res == 1) {
            (*events_count) += ROCS_XGMI_BW_VARIANT__NUM;
        }
    }

    return PAPI_OK;
}

int
handle_derived_events(const char *v_name, int32_t dev, int64_t v_variant, int64_t v_subvariant, int *events_count, ntv_event_t *events)
{
    /* NOTE: derived events are of two types:
     * (a) one rsmi event contains aggregated pieces of data and is thus
     *     split into separate events in the rocm smicomponent;
     * (b) two rsmi events are merged into a single one in the rocm smi
     *     component. An example is the variants set/get which are
     *     both represented by the same native event in PAPI.
     */

    if (strcmp(v_name, "rsmi_dev_pci_throughput_get") == 0) {
        int64_t i;
        for (i = 0; i < ROCS_PCI_THROUGHPUT_VARIANT__NUM; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, i, -1);
            events[*events_count].descr = get_event_descr(v_name, i, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_power_profile_presets_get") == 0) {
        int64_t i;
        for (i = 0; i < ROCS_POWER_PRESETS_VARIANT__NUM; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, i, -1);
            events[*events_count].descr = get_event_descr(v_name, i, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_power_cap_range_get") == 0) {
        int64_t i;
        for (i = 0; i < ROCS_POWER_CAP_RANGE_VARIANT__NUM; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, i, v_subvariant);
            events[*events_count].descr = get_event_descr(v_name, i, v_subvariant);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = v_subvariant;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_ecc_count_get") == 0) {
        int64_t i;
        for (i = 0; i < ROCS_ECC_COUNT_SUBVARIANT__NUM; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, v_variant, i);
            events[*events_count].descr = get_event_descr(v_name, v_variant, i);
            events[*events_count].device = dev;
            events[*events_count].variant = v_variant;
            events[*events_count].subvariant = i;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_pci_bandwidth_get") == 0) {
        if (pcie_table[dev].transfer_rate.num_supported == 0) {
            return ROCS_EVENT_TYPE__DERIVED;
        }
        int64_t i;
        for (i = 0; i <= ROCS_PCI_BW_VARIANT__CURRENT; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, i, -1);
            events[*events_count].descr = get_event_descr(v_name, i, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        int64_t j;
        for (i = ROCS_PCI_BW_VARIANT__CURRENT + 1; i <= ROCS_PCI_BW_VARIANT__LANE_IDX; ++i) {
           for (j = 0; j < pcie_table[dev].transfer_rate.num_supported; ++j) {
               events[*events_count].id = *events_count;
               events[*events_count].name = get_event_name(v_name, dev, i, j);
               events[*events_count].descr = get_event_descr(v_name, i, j);
               events[*events_count].device = dev;
               events[*events_count].variant = i;
               events[*events_count].subvariant = j;
               events[*events_count].mode = ROCS_ACCESS_MODE__READ;
               events[*events_count].open_func_p = get_open_func(v_name);
               events[*events_count].close_func_p = get_close_func(v_name);
               events[*events_count].start_func_p = get_start_func(v_name);
               events[*events_count].stop_func_p = get_stop_func(v_name);
               events[*events_count].access_func_p = get_access_func(v_name);
               htable_insert(htable, events[*events_count].name, &events[*events_count]);
               ++(*events_count);
           }
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_pci_bandwidth_set") == 0) {
        if (pcie_table[dev].transfer_rate.num_supported) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, v_variant, v_subvariant);
            events[*events_count].descr = get_event_descr(v_name, v_variant, v_subvariant);
            events[*events_count].device = dev;
            events[*events_count].variant = v_variant;
            events[*events_count].subvariant = v_subvariant;
            events[*events_count].mode = ROCS_ACCESS_MODE__WRITE;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_gpu_clk_freq_get") == 0) {
        int64_t i;
        int table_id = dev * ROCS_GPU_CLK_FREQ_VARIANT__NUM + v_variant;
        for (i = 0; i < ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM; ++i) {
            if (freq_table[table_id].num_supported) {
                events[*events_count].id = *events_count;
                events[*events_count].name = get_event_name(v_name, dev, v_variant, i);
                events[*events_count].descr = get_event_descr(v_name, v_variant, i);
                events[*events_count].device = dev;
                events[*events_count].variant = v_variant;
                events[*events_count].subvariant = i;
                events[*events_count].mode = ROCS_ACCESS_MODE__READ;
                events[*events_count].open_func_p = get_open_func(v_name);
                events[*events_count].close_func_p = get_close_func(v_name);
                events[*events_count].start_func_p = get_start_func(v_name);
                events[*events_count].stop_func_p = get_stop_func(v_name);
                events[*events_count].access_func_p = get_access_func(v_name);
                htable_insert(htable, events[*events_count].name, &events[*events_count]);
                ++(*events_count);
            }
        }

        for (i = 0; i < freq_table[table_id].num_supported; ++i) {
            int idx = ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM + i;
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, v_variant, idx);
            events[*events_count].descr = get_event_descr(v_name, v_variant, idx);
            events[*events_count].device = dev;
            events[*events_count].variant = v_variant;
            events[*events_count].subvariant = idx;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    if (strcmp(v_name, "rsmi_dev_gpu_clk_freq_set") == 0) {
        int table_id = dev * ROCS_GPU_CLK_FREQ_VARIANT__NUM + v_variant;
        if (freq_table[table_id].num_supported) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name(v_name, dev, v_variant, -1);
            events[*events_count].descr = get_event_descr(v_name, v_variant, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = v_variant;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__WRITE;
            events[*events_count].open_func_p = get_open_func(v_name);
            events[*events_count].close_func_p = get_close_func(v_name);
            events[*events_count].start_func_p = get_start_func(v_name);
            events[*events_count].stop_func_p = get_stop_func(v_name);
            events[*events_count].access_func_p = get_access_func(v_name);
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_fan_speed_count[PAPI_ROCMSMI_MAX_DEV_COUNT];
    if (strcmp(v_name, "rsmi_dev_fan_speed_get") == 0 || strcmp(v_name, "rsmi_dev_fan_speed_set") == 0) {
        if (rsmi_dev_fan_speed_count[dev] == 0) {
            rsmi_dev_fan_speed_count[dev] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        if (strcmp(v_name, "rsmi_dev_fan_speed_set") == 0) {
            /* Set overwrites Get, update event description */
            papi_free(events[rsmi_dev_fan_speed_count[dev]].descr);
            events[rsmi_dev_fan_speed_count[dev]].descr = get_event_descr(v_name, -1, -1);
            events[rsmi_dev_fan_speed_count[dev]].mode = ROCS_ACCESS_MODE__RDWR;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_power_cap_count[PAPI_ROCMSMI_MAX_DEV_COUNT][PAPI_ROCMSMI_MAX_SUBVAR];
    if (strcmp(v_name, "rsmi_dev_power_cap_get") == 0 || strcmp(v_name, "rsmi_dev_power_cap_set") == 0) {
        if (rsmi_dev_power_cap_count[dev][v_subvariant] == 0) {
            rsmi_dev_power_cap_count[dev][v_subvariant] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        if (strcmp(v_name, "rsmi_dev_power_cap_set") == 0) {
            /* Set overwrites Get, update event description */
            papi_free(events[rsmi_dev_power_cap_count[dev][v_subvariant]].descr);
            events[rsmi_dev_power_cap_count[dev][v_subvariant]].descr = get_event_descr(v_name, -1, -1);
            events[rsmi_dev_power_cap_count[dev][v_subvariant]].mode = ROCS_ACCESS_MODE__RDWR;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    static int rsmi_dev_perf_level_count[PAPI_ROCMSMI_MAX_DEV_COUNT];
    if (strncmp(v_name, "rsmi_dev_perf_level", strlen("rsmi_dev_perf_level")) == 0) {
        if (rsmi_dev_perf_level_count[dev] == 0) {
            rsmi_dev_perf_level_count[dev] = *events_count;
            return ROCS_EVENT_TYPE__NATIVE;
        }

        if (strcmp(v_name, "rsmi_dev_perf_level_set") == 0) {
            /* Set overwrites Get, update event description */
            papi_free(events[rsmi_dev_perf_level_count[dev]].descr);
            events[rsmi_dev_perf_level_count[dev]].descr = get_event_descr(v_name, -1, -1);
            events[rsmi_dev_perf_level_count[dev]].mode = ROCS_ACCESS_MODE__RDWR;
        }

        return ROCS_EVENT_TYPE__DERIVED;
    }

    return ROCS_EVENT_TYPE__NATIVE;
}

int
handle_xgmi_events(int32_t dev, int *events_count, ntv_event_t *events)
{
    int i;
    rsmi_status_t status;

    status = rsmi_dev_counter_group_supported_p(dev, RSMI_EVNT_GRP_XGMI);
    if (status == RSMI_STATUS_SUCCESS) {
        for (i = RSMI_EVNT_XGMI_FIRST; i < RSMI_EVNT_XGMI_LAST; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name("rsmi_dev_xgmi_evt_get", dev, i, -1);
            events[*events_count].descr = get_event_descr("rsmi_dev_xgmi_evt_get", i, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].close_func_p = get_close_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].start_func_p = get_start_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].stop_func_p = get_stop_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].access_func_p = get_access_func("rsmi_dev_xgmi_evt_get");
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }
    }

    status = rsmi_dev_counter_group_supported_p(dev, RSMI_EVNT_GRP_XGMI_DATA_OUT);
    if (status == RSMI_STATUS_SUCCESS) {
        for (i = RSMI_EVNT_XGMI_DATA_OUT_FIRST; i < RSMI_EVNT_XGMI_DATA_OUT_LAST; ++i) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name("rsmi_dev_xgmi_evt_get", dev, i, -1);
            events[*events_count].descr = get_event_descr("rsmi_dev_xgmi_evt_get", i, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = i;
            events[*events_count].subvariant = -1;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].close_func_p = get_close_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].start_func_p = get_start_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].stop_func_p = get_stop_func("rsmi_dev_xgmi_evt_get");
            events[*events_count].access_func_p = get_access_func("rsmi_dev_xgmi_evt_get");
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }
    }

    int j;
    for (i = 0; i < device_count; ++i) {
        if (i == dev) {
            continue;
        }
        rsmi_status_t status;
        int res = 0;
        status = rsmi_is_P2P_accessible_p((uint32_t) i, (uint32_t) dev, &res);
        if (status != RSMI_STATUS_SUCCESS) {
            break;
        }
        if (res == 0) {
            continue;
        }
        uint64_t min, max;
        status = rsmi_minmax_bandwidth_get_p((uint32_t) dev, (uint32_t) i, &min, &max);
        if (status != RSMI_STATUS_SUCCESS) {
            break;
        }
        for (j = 0; j < ROCS_XGMI_BW_VARIANT__NUM; ++j) {
            events[*events_count].id = *events_count;
            events[*events_count].name = get_event_name("rsmi_dev_xgmi_bw_get", dev, (int64_t) j, (int64_t) i);
            events[*events_count].descr = get_event_descr("rsmi_dev_xgmi_bw_get", (int64_t) j, -1);
            events[*events_count].device = dev;
            events[*events_count].variant = j;
            events[*events_count].subvariant = i;
            events[*events_count].mode = ROCS_ACCESS_MODE__READ;
            events[*events_count].open_func_p = get_open_func("rsmi_dev_xgmi_bw_get");
            events[*events_count].close_func_p = get_close_func("rsmi_dev_xgmi_bw_get");
            events[*events_count].start_func_p = get_start_func("rsmi_dev_xgmi_bw_get");
            events[*events_count].stop_func_p = get_stop_func("rsmi_dev_xgmi_bw_get");
            events[*events_count].access_func_p = get_access_func("rsmi_dev_xgmi_bw_get");
            htable_insert(htable, events[*events_count].name, &events[*events_count]);
            ++(*events_count);
        }
    }

    return PAPI_OK;
}

char *
get_event_name(const char *name, int32_t dev, int64_t variant, int64_t subvariant)
{
    char event_name_str[PAPI_MAX_STR_LEN] = { 0 };

    if (strcmp(name, "rsmi_dev_count") == 0) {
        return strdup("NUMDevices");
    } else if (strcmp(name, "rsmi_lib_version") == 0) {
        return strdup("rsmi_version");
    } else if (strcmp(name, "rsmi_dev_driver_version_str_get") == 0) {
        return strdup("driver_version_str");
    } else if (strcmp(name, "rsmi_dev_id_get") == 0) {
        sprintf(event_name_str, "device_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_subsystem_vendor_id_get") == 0) {
        sprintf(event_name_str, "subsystem_vendor_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_vendor_id_get") == 0) {
        sprintf(event_name_str, "vendor_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_unique_id_get") == 0) {
        sprintf(event_name_str, "unique_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_subsystem_id_get") == 0) {
        sprintf(event_name_str, "subsystem_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_drm_render_minor_get") == 0) {
        sprintf(event_name_str, "drm_render_minor:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_overdrive_level_get") == 0) {
        sprintf(event_name_str, "overdrive_level:device=%i", dev);
    } else if (strncmp(name, "rsmi_dev_perf_level", strlen("rsmi_dev_perf_level")) == 0) {
        sprintf(event_name_str, "perf_level:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_memory_total_get") == 0) {
        switch (variant) {
            case RSMI_MEM_TYPE_VRAM:
                sprintf(event_name_str, "mem_total_VRAM:device=%i", dev);
                break;
            case RSMI_MEM_TYPE_VIS_VRAM:
                sprintf(event_name_str, "mem_total_VIS_VRAM:device=%i", dev);
                break;
            case RSMI_MEM_TYPE_GTT:
                sprintf(event_name_str, "mem_total_GTT:device=%i", dev);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_memory_usage_get") == 0) {
        switch (variant) {
            case RSMI_MEM_TYPE_VRAM:
                sprintf(event_name_str, "mem_usage_VRAM:device=%i", dev);
                break;
            case RSMI_MEM_TYPE_VIS_VRAM:
                sprintf(event_name_str, "mem_usage_VIS_VRAM:device=%i", dev);
                break;
            case RSMI_MEM_TYPE_GTT:
                sprintf(event_name_str, "mem_usage_GTT:device=%i", dev);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_busy_percent_get") == 0) {
        sprintf(event_name_str, "busy_percent:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_memory_busy_percent_get") == 0) {
        sprintf(event_name_str, "memory_busy_percent:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_pci_id_get") == 0) {
        sprintf(event_name_str, "pci_id:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_pci_replay_counter_get") == 0) {
        sprintf(event_name_str, "pci_replay_counter:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_pci_throughput_get") == 0) {
        switch (variant) {
            case ROCS_PCI_THROUGHPUT_VARIANT__SENT:
                sprintf(event_name_str, "pci_throughput_sent:device=%i", dev);
                break;
            case ROCS_PCI_THROUGHPUT_VARIANT__RECEIVED:
                sprintf(event_name_str, "pci_throughput_received:device=%i", dev);
                break;
            case ROCS_PCI_THROUGHPUT_VARIANT__MAX_PACKET_SIZE:
                sprintf(event_name_str, "pci_max_packet_size:device=%i", dev);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_power_profile_presets_get") == 0) {
        switch (variant) {
            case ROCS_POWER_PRESETS_VARIANT__COUNT:
                sprintf(event_name_str, "power_profiler_presets:device=%i:count", dev);
                break;
            case ROCS_POWER_PRESETS_VARIANT__AVAIL_PROFILES:
                sprintf(event_name_str, "power_profiler_presets:device=%i:avail_profiles", dev);
                break;
            case ROCS_POWER_PRESETS_VARIANT__CURRENT:
                sprintf(event_name_str, "power_profiler_presets:device=%i:current", dev);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_power_profile_set") == 0) {
        sprintf(event_name_str, "power_profile_set:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_fan_reset") == 0) {
        sprintf(event_name_str, "fan_reset:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_fan_rpms_get") == 0) {
        sprintf(event_name_str, "fan_rpms:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_fan_speed_max_get") == 0) {
        sprintf(event_name_str, "fan_speed_max:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_fan_speed_get") == 0 || strcmp(name, "rsmi_dev_fan_speed_set") == 0) {
        sprintf(event_name_str, "fan_speed:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_power_ave_get") == 0) {
        sprintf(event_name_str, "power_average:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_power_cap_get") == 0 || strcmp(name, "rsmi_dev_power_cap_set") == 0) {
        sprintf(event_name_str, "power_cap:device=%i:sensor=%i", dev, (int) subvariant);
    } else if (strcmp(name, "rsmi_dev_power_cap_range_get") == 0) {
        switch (variant) {
            case ROCS_POWER_CAP_RANGE_VARIANT__MIN:
                sprintf(event_name_str, "power_cap_range_min:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case ROCS_POWER_CAP_RANGE_VARIANT__MAX:
                sprintf(event_name_str, "power_cap_range_max:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_temp_metric_get") == 0) {
        switch (variant) {
            case RSMI_TEMP_CURRENT:
                sprintf(event_name_str, "temp_current:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_MAX:
                sprintf(event_name_str, "temp_max:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_MIN:
                sprintf(event_name_str, "temp_min:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_MAX_HYST:
                sprintf(event_name_str, "temp_max_hyst:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_MIN_HYST:
                sprintf(event_name_str, "temp_min_hyst:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_CRITICAL:
                sprintf(event_name_str, "temp_critical:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_CRITICAL_HYST:
                sprintf(event_name_str, "temp_critical_hyst:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_EMERGENCY:
                sprintf(event_name_str, "temp_emergency:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_EMERGENCY_HYST:
                sprintf(event_name_str, "temp_emergency_hyst:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_CRIT_MIN:
                sprintf(event_name_str, "temp_crit_min:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_CRIT_MIN_HYST:
                sprintf(event_name_str, "temp_crit_min_hyst:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_OFFSET:
                sprintf(event_name_str, "temp_offset:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_LOWEST:
                sprintf(event_name_str, "temp_lowest:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            case RSMI_TEMP_HIGHEST:
                sprintf(event_name_str, "temp_highest:device=%i:sensor=%i", dev, (int) subvariant);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_firmware_version_get") == 0) {
        switch (variant) {
            case RSMI_FW_BLOCK_ASD:
                sprintf(event_name_str, "firmware_version:device=%i:block=ASD", dev);
                break;
            case RSMI_FW_BLOCK_CE:
                sprintf(event_name_str, "firmware_version:device=%i:block=CE", dev);
                break;
            case RSMI_FW_BLOCK_DMCU:
                sprintf(event_name_str, "firmware_version:device=%i:block=DMCU", dev);
                break;
            case RSMI_FW_BLOCK_MC:
                sprintf(event_name_str, "firmware_version:device=%i:block=MC", dev);
                break;
            case RSMI_FW_BLOCK_ME:
                sprintf(event_name_str, "firmware_version:device=%i:block=ME", dev);
                break;
            case RSMI_FW_BLOCK_MEC:
                sprintf(event_name_str, "firmware_version:device=%i:block=MEC", dev);
                break;
            case RSMI_FW_BLOCK_MEC2:
                sprintf(event_name_str, "firmware_version:device=%i:block=MEC2", dev);
                break;
            case RSMI_FW_BLOCK_PFP:
                sprintf(event_name_str, "firmware_version:device=%i:block=PFP", dev);
                break;
            case RSMI_FW_BLOCK_RLC:
                sprintf(event_name_str, "firmware_version:device=%i:block=RLC", dev);
                break;
            case RSMI_FW_BLOCK_RLC_SRLC:
                sprintf(event_name_str, "firmware_version:device=%i:block=SRLC", dev);
                break;
            case RSMI_FW_BLOCK_RLC_SRLG:
                sprintf(event_name_str, "firmware_version:device=%i:block=SRLG", dev);
                break;
            case RSMI_FW_BLOCK_RLC_SRLS:
                sprintf(event_name_str, "firmware_version:device=%i:block=SRLS", dev);
                break;
            case RSMI_FW_BLOCK_SDMA:
                sprintf(event_name_str, "firmware_version:device=%i:block=SDMA", dev);
                break;
            case RSMI_FW_BLOCK_SDMA2:
                sprintf(event_name_str, "firmware_version:device=%i:block=SDMA2", dev);
                break;
            case RSMI_FW_BLOCK_SMC:
                sprintf(event_name_str, "firmware_version:device=%i:block=SMC", dev);
                break;
            case RSMI_FW_BLOCK_SOS:
                sprintf(event_name_str, "firmware_version:device=%i:block=SOS", dev);
                break;
            case RSMI_FW_BLOCK_TA_RAS:
                sprintf(event_name_str, "firmware_version:device=%i:block=RAS", dev);
                break;
            case RSMI_FW_BLOCK_TA_XGMI:
                sprintf(event_name_str, "firmware_version:device=%i:block=XGMI", dev);
                break;
            case RSMI_FW_BLOCK_UVD:
                sprintf(event_name_str, "firmware_version:device=%i:block=UVD", dev);
                break;
            case RSMI_FW_BLOCK_VCE:
                sprintf(event_name_str, "firmware_version:device=%i:block=VCE", dev);
                break;
            case RSMI_FW_BLOCK_VCN:
                sprintf(event_name_str, "firmware_version:device=%i:block=VCN", dev);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_ecc_count_get") == 0) {
        const char *block = NULL;
        switch (variant) {
            case RSMI_GPU_BLOCK_UMC:
                block = "UMC";
                break;
            case RSMI_GPU_BLOCK_SDMA:
                block = "SDMA";
                break;
            case RSMI_GPU_BLOCK_GFX:
                block = "GFX";
                break;
            case RSMI_GPU_BLOCK_MMHUB:
                block = "MMHUB";
                break;
            case RSMI_GPU_BLOCK_ATHUB:
                block = "ATHUB";
                break;
            case RSMI_GPU_BLOCK_PCIE_BIF:
                block = "PCIE_BIF";
                break;
            case RSMI_GPU_BLOCK_HDP:
                block = "HDP";
                break;
            case RSMI_GPU_BLOCK_XGMI_WAFL:
                block = "XGMI_WAFL";
                break;
            case RSMI_GPU_BLOCK_DF:
                block = "DF";
                break;
            case RSMI_GPU_BLOCK_SMN:
                block = "SMN";
                break;
            case RSMI_GPU_BLOCK_SEM:
                block = "SEM";
                break;
            case RSMI_GPU_BLOCK_MP0:
                block = "MP0";
                break;
            case RSMI_GPU_BLOCK_MP1:
                block = "MP1";
                break;
            case RSMI_GPU_BLOCK_FUSE:
                block = "FUSE";
                break;
            default:
                return NULL;
        }

        switch (subvariant) {
            case ROCS_ECC_COUNT_SUBVARIANT__CORRECTABLE:
                sprintf(event_name_str, "ecc_count_correctable:device=%i:block=%s", dev, block);
                break;
            case ROCS_ECC_COUNT_SUBVARIANT__UNCORRECTABLE:
                sprintf(event_name_str, "ecc_count_uncorrectable:device=%i:block=%s", dev, block);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_ecc_enabled_get") == 0) {
        sprintf(event_name_str, "ecc_enabled_get:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_ecc_status_get") == 0) {
        const char *block = NULL;
        switch (variant) {
            case RSMI_GPU_BLOCK_UMC:
                block = "UMC";
                break;
            case RSMI_GPU_BLOCK_SDMA:
                block = "SDMA";
                break;
            case RSMI_GPU_BLOCK_GFX:
                block = "GFX";
                break;
            case RSMI_GPU_BLOCK_MMHUB:
                block = "MMHUB";
                break;
            case RSMI_GPU_BLOCK_ATHUB:
                block = "ATHUB";
                break;
            case RSMI_GPU_BLOCK_PCIE_BIF:
                block = "PCIE_BIF";
                break;
            case RSMI_GPU_BLOCK_HDP:
                block = "HDP";
                break;
            case RSMI_GPU_BLOCK_XGMI_WAFL:
                block = "XGMI_WAFL";
                break;
            case RSMI_GPU_BLOCK_DF:
                block = "DF";
                break;
            case RSMI_GPU_BLOCK_SMN:
                block = "SMN";
                break;
            case RSMI_GPU_BLOCK_SEM:
                block = "SEM";
                break;
            case RSMI_GPU_BLOCK_MP0:
                block = "MP0";
                break;
            case RSMI_GPU_BLOCK_MP1:
                block = "MP1";
                break;
            case RSMI_GPU_BLOCK_FUSE:
                block = "FUSE";
                break;
            default:
                return NULL;
        }
        sprintf(event_name_str, "ecc_status:device=%i:block=%s", dev, block);
    } else if (strcmp(name, "rsmi_dev_gpu_clk_freq_get") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_GPU_CLK_FREQ_VARIANT__SYSTEM:
                variant_str = "System";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DATA_FABRIC:
                variant_str = "DataFabric";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DISPLAY_ENGINE:
                variant_str = "DisplayEngine";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__SOC:
                variant_str = "SOC";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__MEMORY:
                variant_str = "Memory";
                break;
            default:
                return NULL;
        }

        int idx;
        const char *subvariant_str = NULL;
        switch (subvariant) {
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__COUNT:
                subvariant_str = "count";
                break;
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__CURRENT:
                subvariant_str = "current";
                break;
            default:
                idx = subvariant - ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM;
        }

        if (subvariant <= ROCS_GPU_CLK_FREQ_SUBVARIANT__CURRENT) {
            sprintf(event_name_str, "gpu_clk_freq_%s:device=%i:%s", variant_str, dev, subvariant_str);
        } else {
            sprintf(event_name_str, "gpu_clk_freq_%s:device=%i:idx=%i", variant_str, dev, idx);
        }
    } else if (strcmp(name, "rsmi_dev_gpu_clk_freq_set") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_GPU_CLK_FREQ_VARIANT__SYSTEM:
                variant_str = "System";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DATA_FABRIC:
                variant_str = "DataFabric";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DISPLAY_ENGINE:
                variant_str = "DisplayEngine";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__SOC:
                variant_str = "SOC";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__MEMORY:
                variant_str = "Memory";
                break;
            default:
                return NULL;
        }

        sprintf(event_name_str, "gpu_clk_freq_%s:device=%i:mask", variant_str, dev);
    } else if (strcmp(name, "rsmi_dev_pci_bandwidth_get") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_PCI_BW_VARIANT__COUNT:
                variant_str = "count";
                break;
            case ROCS_PCI_BW_VARIANT__CURRENT:
                variant_str = "current";
                break;
            case ROCS_PCI_BW_VARIANT__RATE_IDX:
                variant_str = "rate_idx";
                break;
            case ROCS_PCI_BW_VARIANT__LANE_IDX:
                variant_str = "lane_idx";
                break;
            default:
                return NULL;
        }

        if (variant <= ROCS_PCI_BW_VARIANT__CURRENT) {
            sprintf(event_name_str, "pci_bandwidth_rate:device=%i:%s", dev, variant_str);
        } else {
            sprintf(event_name_str, "pci_bandwidth_rate:device=%i:%s=%i", dev, variant_str, (int) subvariant);
        }
    } else if (strcmp(name, "rsmi_dev_pci_bandwidth_set") == 0) {
        sprintf(event_name_str, "pci_bandwidth_rate:device=%i:mask", dev);
    } else if (strcmp(name, "rsmi_dev_brand_get") == 0) {
        sprintf(event_name_str, "device_brand:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_name_get") == 0) {
        sprintf(event_name_str, "device_name:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_serial_number_get") == 0) {
        sprintf(event_name_str, "device_serial_number:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_subsystem_name_get") == 0) {
        sprintf(event_name_str, "device_subsystem_name:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_vbios_version_get") == 0) {
        sprintf(event_name_str, "vbios_version:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_vendor_name_get") == 0) {
        sprintf(event_name_str, "vendor_name:device=%i", dev);
    } else if (strcmp(name, "rsmi_dev_xgmi_evt_get") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_XGMI_VARIANT__MI50_0_NOP_TX:
                variant_str = "nop_sent_to_neighbor0";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_REQUEST_TX:
                variant_str = "req_sent_to_neighbor0";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_RESPONSE_TX:
                variant_str = "res_sent_to_neighbor0";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_BEATS_TX:
                variant_str = "data_beats_sent_to_neighbor0";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_NOP_TX:
                variant_str = "nop_sent_to_neighbor1";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_REQUEST_TX:
                variant_str = "req_sent_to_neighbor1";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_RESPONSE_TX:
                variant_str = "res_sent_to_neighbor1";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_BEATS_TX:
                variant_str = "data_beats_sent_to_neighbor1";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_0:
                variant_str = "data_beats_sent_to_neighbor0";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_1:
                variant_str = "data_beats_sent_to_neighbor1";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_2:
                variant_str = "data_beats_sent_to_neighbor2";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_3:
                variant_str = "data_beats_sent_to_neighbor3";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_4:
                variant_str = "data_beats_sent_to_neighbor4";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_5:
                variant_str = "data_beats_sent_to_neighbor5";
                break;
            default:
                return NULL;
        }
        sprintf(event_name_str, "xgmi_%s:device=%i", variant_str, dev);
    } else if (strcmp(name, "rsmi_dev_xgmi_bw_get") == 0) {
        switch (variant) {
            case ROCS_XGMI_BW_VARIANT__MIN:
                sprintf(event_name_str, "min_xgmi_internode_bw:device=%i:target=%i", dev, (int) subvariant);
                break;
            case ROCS_XGMI_BW_VARIANT__MAX:
                sprintf(event_name_str, "max_xgmi_internode_bw:device=%i:target=%i", dev, (int) subvariant);
                break;
            default:
                return NULL;
        }
    } else {
        return NULL;
    }

    return strdup(event_name_str);
}

char *
get_event_descr(const char *name, int64_t variant, int64_t subvariant)
{
    char event_descr_str[PAPI_MAX_STR_LEN] = { 0 };

    if (strcmp(name, "rsmi_dev_count") == 0) {
        return strdup("Number of Devices which have monitors, accessible by rocm_smi.");
    } else if (strcmp(name, "rsmi_lib_version") == 0) {
        return strdup("Version of RSMI lib; 0x0000MMMMmmmmpppp Major, Minor, Patch.");
    } else if (strcmp(name, "rsmi_dev_driver_version_str_get") == 0) {
        return strdup("Returns char* to z-terminated driver version string; do not free().");
    } else if (strcmp(name, "rsmi_dev_id_get") == 0) {
        return strdup("Vendor supplied device id number. May be shared by same model devices; see pci_id for a unique identifier.");
    } else if (strcmp(name, "rsmi_dev_subsystem_vendor_id_get") == 0) {
        return strdup("System vendor id number.");
    } else if (strcmp(name, "rsmi_dev_vendor_id_get") == 0) {
        return strdup("Vendor id number.");
    } else if (strcmp(name, "rsmi_dev_unique_id_get") == 0) {
        return strdup("Unique id for device.");
    } else if (strcmp(name, "rsmi_dev_subsystem_id_get") == 0) {
        return strdup("Subsystem id number.");
    } else if (strcmp(name, "rsmi_dev_drm_render_minor_get") == 0) {
        return strdup("DRM Minor Number associated with this device.");
    } else if (strcmp(name, "rsmi_dev_overdrive_level_get") == 0) {
        return strdup("Overdriver Level \% for device, 0 to 20, max overclocked permitted. Read Only.");
    } else if (strcmp(name, "rsmi_dev_perf_level_get") == 0) {
        sprintf(event_descr_str, "PowerPlay Performance Level; Read Only, enum rsmi_dev_perf_level_t [0-%i], see ROCm_SMI_Manual for details.",
                RSMI_DEV_PERF_LEVEL_LAST);
    } else if (strcmp(name, "rsmi_dev_perf_level_set") == 0) {
        sprintf(event_descr_str, "PowerPlay Performance Level; Read/Write, enum rsmi_dev_perf_level_t [0-%i], see ROCm_SMI_Manual for details.",
                RSMI_DEV_PERF_LEVEL_LAST);
    } else if (strcmp(name, "rsmi_dev_memory_total_get") == 0) {
        switch (variant) {
            case RSMI_MEM_TYPE_VRAM:
                sprintf(event_descr_str, "Total VRAM memory.");
                break;
            case RSMI_MEM_TYPE_VIS_VRAM:
                sprintf(event_descr_str, "Total Visible VRAM memory.");
                break;
            case RSMI_MEM_TYPE_GTT:
                sprintf(event_descr_str, "Total Visible GTT (Graphics Translation Table) memory, aka GART memory.");
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_memory_usage_get") == 0) {
        switch (variant) {
            case RSMI_MEM_TYPE_VRAM:
                sprintf(event_descr_str, "VRAM memory in use.");
                break;
            case RSMI_MEM_TYPE_VIS_VRAM:
                sprintf(event_descr_str, "Visible VRAM memory in use.");
                break;
            case RSMI_MEM_TYPE_GTT:
                sprintf(event_descr_str, "(Graphic Translation Table) memory in use (aka GART memory).");
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_busy_percent_get") == 0) {
        return strdup("Percentage of time the device was busy doing any processing.");
    } else if (strcmp(name, "rsmi_dev_memory_busy_percent_get") == 0) {
        return strdup("Percentage_of time any device memory is being used.");
    } else if (strcmp(name, "rsmi_dev_pci_id_get") == 0) {
        return strdup("BDF (Bus/Device/Function) ID, unique per device.");
    } else if (strcmp(name, "rsmi_dev_pci_replay_counter_get") == 0) {
        return strdup("Sum of the number of NAK's received by the GPU and the NAK's generated by the GPU.");
    } else if (strcmp(name, "rsmi_dev_pci_throughput_get") == 0) {
        switch (variant) {
            case ROCS_PCI_THROUGHPUT_VARIANT__SENT:
                return strdup("Throughput on PCIe traffic, bytes/second sent.");
            case ROCS_PCI_THROUGHPUT_VARIANT__RECEIVED:
                return strdup("Throughput on PCIe traffic, bytes/second received.");
            case ROCS_PCI_THROUGHPUT_VARIANT__MAX_PACKET_SIZE:
                return strdup("Maximum PCIe packet size.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_power_profile_presets_get") == 0) {
        switch (variant) {
            case ROCS_POWER_PRESETS_VARIANT__COUNT:
                return strdup("Number of power profile presets available. See ROCM_SMI Manual for details.");
            case ROCS_POWER_PRESETS_VARIANT__AVAIL_PROFILES:
                return strdup("Bit mask for available power profile presets. See ROCM_SMI Manual for details.");
            case ROCS_POWER_PRESETS_VARIANT__CURRENT:
                return strdup("Bit mask for current power profile preset. Read/Write. See ROCM_SMI Manual for details.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_power_profile_set") == 0) {
        return strdup("Write Only, set the power profile to one of the available masks. See ROCM_SMI Manual for details.");
    } else if (strcmp(name, "rsmi_dev_fan_reset") == 0) {
        return strdup("Fan Reset. Write Only, data value is ignored.");
    } else if (strcmp(name, "rsmi_dev_fan_rpms_get") == 0) {
        return strdup("Current fan speed in RPMs (Rotations Per Minute).");
    } else if (strcmp(name, "rsmi_dev_fan_speed_max_get") == 0) {
        return strdup("Maximum possible fan speed in RPMs (Rotations Per Minute).");
    } else if (strcmp(name, "rsmi_dev_fan_speed_get") == 0) {
        return strdup("Current fan speed in RPMs (Rotations Per Minute), Read Only, result [0-255].");
    } else if (strcmp(name, "rsmi_dev_fan_speed_set") == 0) {
        return strdup("Current fan speed in RPMs (Rotations Per Minute), Read/Write, Write must be <= MAX (see fan_speed_max event), arg in [0-255].");
    } else if (strcmp(name, "rsmi_dev_power_ave_get") == 0) {
        return strdup("Current Average Power consumption in microwatts. Requires root privileges.");
    } else if (strcmp(name, "rsmi_dev_power_cap_get") == 0) {
        return strdup("Power cap in microwatts. Read Only. Between min/max (see power_cap_range_min/max). May require root privileges.");
    } else if (strcmp(name, "rsmi_dev_power_cap_set") == 0) {
        return strdup("Power cap in microwatts. Read/Write. Between min/max (see power_cap_range_min/max). May require root privileges.");
    } else if (strcmp(name, "rsmi_dev_power_cap_range_get") == 0) {
        switch (variant) {
            case ROCS_POWER_CAP_RANGE_VARIANT__MIN:
                return strdup("Power cap Minimum settable value, in microwatts.");
            case ROCS_POWER_CAP_RANGE_VARIANT__MAX:
                return strdup("Power cap Maximim settable value, in microwatts.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_temp_metric_get") == 0) {
        switch (variant) {
            case RSMI_TEMP_CURRENT:
                return strdup("Temperature current value, millidegrees Celsius.");
            case RSMI_TEMP_MAX:
                return strdup("Temperature maximum value, millidegrees Celsius.");
            case RSMI_TEMP_MIN:
                return strdup("Temperature minimum value, millidegrees Celsius.");
            case RSMI_TEMP_MAX_HYST:
                return strdup("Temperature hysteresis value for max limit, millidegrees Celsius.");
            case RSMI_TEMP_MIN_HYST:
                return strdup("Temperature hysteresis value for min limit, millidegrees Celsius.");
            case RSMI_TEMP_CRITICAL:
                return strdup("Temperature critical max value, typical > temp_max, millidegrees Celsius.");
            case RSMI_TEMP_CRITICAL_HYST:
                return strdup("Temperature hysteresis value for critical limit, millidegrees Celsius.");
            case RSMI_TEMP_EMERGENCY:
                return strdup("Temperature emergency max for chips supporting more than two upper temp limits, millidegrees Celsius.");
            case RSMI_TEMP_EMERGENCY_HYST:
                return strdup("Temperature hysteresis value for emergency limit, millidegrees Celsius.");
            case RSMI_TEMP_CRIT_MIN:
                return strdup("Temperature critical min value, typical < temp_min, millidegrees Celsius.");
            case RSMI_TEMP_CRIT_MIN_HYST:
                return strdup("Temperature hysteresis value for critical min limit, millidegrees Celsius.");
            case RSMI_TEMP_OFFSET:
                return strdup("Temperature offset added to temp reading by the chip, millidegrees Celsius.");
            case RSMI_TEMP_LOWEST:
                return strdup("Temperature historical minimum, millidegrees Celsius.");
            case RSMI_TEMP_HIGHEST:
                return strdup("Temperature historical maximum, millidegrees Celsius.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_firmware_version_get") == 0) {
        switch (variant) {
            case RSMI_FW_BLOCK_ASD:
                return strdup("Firmware Version Block ASD.");
            case RSMI_FW_BLOCK_CE:
                return strdup("Firmware Version Block CE.");
            case RSMI_FW_BLOCK_DMCU:
                return strdup("Firmware Version Block DMCU.");
            case RSMI_FW_BLOCK_MC:
                return strdup("Firmware Version Block MC.");
            case RSMI_FW_BLOCK_ME:
                return strdup("Firmware Version Block ME.");
            case RSMI_FW_BLOCK_MEC:
                return strdup("Firmware Version Block MEC.");
            case RSMI_FW_BLOCK_MEC2:
                return strdup("Firmware Version Block MEC2.");
            case RSMI_FW_BLOCK_PFP:
                return strdup("Firmware Version Block PFP.");
            case RSMI_FW_BLOCK_RLC:
                return strdup("Firmware Version Block RLC.");
            case RSMI_FW_BLOCK_RLC_SRLC:
                return strdup("Firmware Version Block SRLC.");
            case RSMI_FW_BLOCK_RLC_SRLG:
                return strdup("Firmware Version Block SRLG.");
            case RSMI_FW_BLOCK_RLC_SRLS:
                return strdup("Firmware Version Block SRLS.");
            case RSMI_FW_BLOCK_SDMA:
                return strdup("Firmware Version Block SDMA.");
            case RSMI_FW_BLOCK_SDMA2:
                return strdup("Firmware Version Block SDMA2.");
            case RSMI_FW_BLOCK_SMC:
                return strdup("Firmware Version Block SMC.");
            case RSMI_FW_BLOCK_SOS:
                return strdup("Firmware Version Block SOS.");
            case RSMI_FW_BLOCK_TA_RAS:
                return strdup("Firmware Version Block RAS.");
            case RSMI_FW_BLOCK_TA_XGMI:
                return strdup("Firmware Version Block XGMI.");
            case RSMI_FW_BLOCK_UVD:
                return strdup("Firmware Version Block UVD.");
            case RSMI_FW_BLOCK_VCE:
                return strdup("Firmware Version Block VCE.");
            case RSMI_FW_BLOCK_VCN:
                return strdup("Firmware Version Block VCN.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_ecc_count_get") == 0) {
        const char *block = NULL;
        switch (variant) {
            case RSMI_GPU_BLOCK_UMC:
                block = "UMC";
                break;
            case RSMI_GPU_BLOCK_SDMA:
                block = "SDMA";
                break;
            case RSMI_GPU_BLOCK_GFX:
                block = "GFX";
                break;
            case RSMI_GPU_BLOCK_MMHUB:
                block = "MMHUB";
                break;
            case RSMI_GPU_BLOCK_ATHUB:
                block = "ATHUB";
                break;
            case RSMI_GPU_BLOCK_PCIE_BIF:
                block = "PCIE_BIF";
                break;
            case RSMI_GPU_BLOCK_HDP:
                block = "HDP";
                break;
            case RSMI_GPU_BLOCK_XGMI_WAFL:
                block = "XGMI_WAFL";
                break;
            case RSMI_GPU_BLOCK_DF:
                block = "DF";
                break;
            case RSMI_GPU_BLOCK_SMN:
                block = "SMN";
                break;
            case RSMI_GPU_BLOCK_SEM:
                block = "SEM";
                break;
            case RSMI_GPU_BLOCK_MP0:
                block = "MP0";
                break;
            case RSMI_GPU_BLOCK_MP1:
                block = "MP1";
                break;
            case RSMI_GPU_BLOCK_FUSE:
                block = "FUSE";
                break;
            default:
                return NULL;
        }

        switch (subvariant) {
            case ROCS_ECC_COUNT_SUBVARIANT__CORRECTABLE:
                sprintf(event_descr_str, "Correctable error count for the GPU Block %s.", block);
                break;
            case ROCS_ECC_COUNT_SUBVARIANT__UNCORRECTABLE:
                sprintf(event_descr_str, "Uncorrectable error count for the GPU Block %s.", block);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_ecc_enabled_get") == 0) {
        return strdup("Bit mask of GPU blocks with ecc error counting enabled.");
    } else if (strcmp(name, "rsmi_dev_ecc_status_get") == 0) {
        switch (variant) {
            case RSMI_GPU_BLOCK_UMC:
                return strdup("ECC Error Status for the GPU Block UMC.");
            case RSMI_GPU_BLOCK_SDMA:
                return strdup("ECC Error Status for the GPU Block SDMA.");
            case RSMI_GPU_BLOCK_GFX:
                return strdup("ECC Error Status for the GPU Block GFX.");
            case RSMI_GPU_BLOCK_MMHUB:
                return strdup("ECC Error Status for the GPU Block MMHUB.");
            case RSMI_GPU_BLOCK_ATHUB:
                return strdup("ECC Error Status for the GPU Block ATHUB.");
            case RSMI_GPU_BLOCK_PCIE_BIF:
                return strdup("ECC Error Status for the GPU Block BIF.");
            case RSMI_GPU_BLOCK_HDP:
                return strdup("ECC Error Status for the GPU Block HDP.");
            case RSMI_GPU_BLOCK_XGMI_WAFL:
                return strdup("ECC Error Status for the GPU Block WAFL.");
            case RSMI_GPU_BLOCK_DF:
                return strdup("ECC Error Status for the GPU Block DF.");
            case RSMI_GPU_BLOCK_SMN:
                return strdup("ECC Error Status for the GPU Block SMN.");
            case RSMI_GPU_BLOCK_SEM:
                return strdup("ECC Error Status for the GPU Block SEM.");
            case RSMI_GPU_BLOCK_MP0:
                return strdup("ECC Error Status for the GPU Block MP0.");
            case RSMI_GPU_BLOCK_MP1:
                return strdup("ECC Error Status for the GPU Block MP1.");
            case RSMI_GPU_BLOCK_FUSE:
                return strdup("ECC Error Status for the GPU Block FUSE.");
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_gpu_clk_freq_get") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_GPU_CLK_FREQ_VARIANT__SYSTEM:
                variant_str = "System";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DATA_FABRIC:
                variant_str = "DataFabric";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DISPLAY_ENGINE:
                variant_str = "DisplayEngine";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__SOC:
                variant_str = "SOC";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__MEMORY:
                variant_str = "Memory";
                break;
            default:
                return NULL;
        }

        int idx;
        switch (subvariant) {
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__COUNT:
                return strdup("Number of frequencies available.");
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__CURRENT:
                return strdup("Current operating frequency.");
            default:
                idx = subvariant - ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM;
        }

        sprintf(event_descr_str, "Returns %s frequency value for supported_table[%u].", variant_str, idx);
    } else if (strcmp(name, "rsmi_dev_gpu_clk_freq_set") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_GPU_CLK_FREQ_VARIANT__SYSTEM:
                variant_str = "System";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DATA_FABRIC:
                variant_str = "DataFabric";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__DISPLAY_ENGINE:
                variant_str = "DisplayEngine";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__SOC:
                variant_str = "SOC";
                break;
            case ROCS_GPU_CLK_FREQ_VARIANT__MEMORY:
                variant_str = "Memory";
                break;
            default:
                return NULL;
        }
        sprintf(event_descr_str, "Write Only. Sets bit mask, 1's for %s frequency value in support table permitted. All 0 mask prohibited.",
                variant_str);
    } else if (strcmp(name, "rsmi_dev_pci_bandwidth_get") == 0) {
        switch (variant) {
            case ROCS_PCI_BW_VARIANT__COUNT:
                return strdup("Number of PCI transfers rates available.");
            case ROCS_PCI_BW_VARIANT__CURRENT:
                return strdup("Current PCI transfer rate.");
            case ROCS_PCI_BW_VARIANT__RATE_IDX:
                sprintf(event_descr_str, "Returns PCI bandwidth rate value from supported_table[%i].", (int) subvariant);
                break;
            case ROCS_PCI_BW_VARIANT__LANE_IDX:
                sprintf(event_descr_str, "Returns PCI bandwidth rate corresponding lane count from supported_table[%i].", (int) subvariant);
                break;
            default:
                return NULL;
        }
    } else if (strcmp(name, "rsmi_dev_pci_bandwidth_set") == 0) {
        return strdup("Write Only. Sets bit mask, 1's for PCI transfer rates in supported_table permitted. All 0 mask prohibited");
    } else if (strcmp(name, "rsmi_dev_brand_get") == 0) {
        return strdup("Returns char* to z-terminated brand string; do not free().");
    } else if (strcmp(name, "rsmi_dev_name_get") == 0) {
        return strdup("Returns char* to z-terminated name string; do not free().");
    } else if (strcmp(name, "rsmi_dev_serial_number_get") == 0) {
        return strdup("Returns char* to z-terminated serial number string; do not free().");
    } else if (strcmp(name, "rsmi_dev_subsystem_name_get") == 0) {
        return strdup("Returns char* to z-terminated subsystem name string; do not free().");
    } else if (strcmp(name, "rsmi_dev_vbios_version_get") == 0) {
        return strdup("Returns char* to z-terminated vbios version string; do not free().");
    } else if (strcmp(name, "rsmi_dev_vendor_name_get") == 0) {
        return strdup("Returns char* to z-terminated vendor name string; do not free().");
    } else if (strcmp(name, "rsmi_dev_xgmi_evt_get") == 0) {
        const char *variant_str = NULL;
        switch (variant) {
            case ROCS_XGMI_VARIANT__MI50_0_NOP_TX:
                variant_str = "NOP operations sent to neightbor 0.";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_REQUEST_TX:
                variant_str = "Outgoing requests to neighbor 0.";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_RESPONSE_TX:
                variant_str = "Outgoing responses sent to neighbor 0.";
                break;
            case ROCS_XGMI_VARIANT__MI50_0_BEATS_TX:
                variant_str = "Data beats sent to neighbor 0.";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_NOP_TX:
                variant_str = "NOP operations sent to neightbor 1.";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_REQUEST_TX:
                variant_str = "Outgoing requests to neighbor 1.";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_RESPONSE_TX:
                variant_str = "Outgoing responses sent to neighbor 1.";
                break;
            case ROCS_XGMI_VARIANT__MI50_1_BEATS_TX:
                variant_str = "Data beats sent to neighbor 1.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_0:
                variant_str = "Data beats sent to neighbor 0.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_1:
                variant_str = "Data beats sent to neighbor 1.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_2:
                variant_str = "Data beats sent to neighbor 2.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_3:
                variant_str = "Data beats sent to neighbor 3.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_4:
                variant_str = "Data beats sent to neighbor 4.";
                break;
            case ROCS_XGMI_VARIANT__MI100_DATA_OUT_5:
                variant_str = "Data beats sent to neighbor 5.";
                break;
            default:
                return NULL;
        }
        sprintf(event_descr_str, "%s", variant_str);
    } else if (strcmp(name, "rsmi_dev_xgmi_bw_get") == 0) {
        switch (variant) {
            case ROCS_XGMI_BW_VARIANT__MIN:
                sprintf(event_descr_str, "%s.", "Minimum bandwidth between devices");
                break;
            case ROCS_XGMI_BW_VARIANT__MAX:
                sprintf(event_descr_str, "%s.", "Maximum bandwidth between devices");
                break;
            default:
                return NULL;
        }
    } else {
        return NULL;
    }

    return strdup(event_descr_str);
}

rocs_access_mode_e
get_access_mode(const char *name)
{
    if (strstr(name, "_get")) {
        return ROCS_ACCESS_MODE__READ;
    } else if (strstr(name, "_put") || strstr(name, "_reset")) {
        return ROCS_ACCESS_MODE__WRITE;
    }
    return ROCS_ACCESS_MODE__READ;
}

open_function_f
get_open_func(const char *name)
{
    int i = 0;
    while (event_function_table[i].name != NULL) {
        if (strcmp(name, event_function_table[i].name) == 0) {
            return event_function_table[i].open_func_p;
        }
        ++i;
    }

    return NULL;
}

close_function_f
get_close_func(const char *name)
{
    int i = 0;
    while (event_function_table[i].name != NULL) {
        if (strcmp(name, event_function_table[i].name) == 0) {
            return event_function_table[i].close_func_p;
        }
        ++i;
    }

    return NULL;
}

start_function_f
get_start_func(const char *name)
{
    int i = 0;
    while (event_function_table[i].name != NULL) {
        if (strcmp(name, event_function_table[i].name) == 0) {
            return event_function_table[i].start_func_p;
        }
        ++i;
    }

    return NULL;
}

stop_function_f
get_stop_func(const char *name)
{
    int i = 0;
    while (event_function_table[i].name != NULL) {
        if (strcmp(name, event_function_table[i].name) == 0) {
            return event_function_table[i].stop_func_p;
        }
        ++i;
    }

    return NULL;
}

access_function_f
get_access_func(const char *name)
{
    int i = 0;
    while (event_function_table[i].name != NULL) {
        if (strcmp(name, event_function_table[i].name) == 0) {
            return event_function_table[i].access_func_p;
        }
        ++i;
    }

    return NULL;
}

int
open_simple(void *arg __attribute__((unused)))
{
    return PAPI_OK;
}

int
close_simple(void *arg __attribute__((unused)))
{
    return PAPI_OK;
}

int
start_simple(void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;
    event->value = 0;
    return PAPI_OK;
}

int
stop_simple(void *arg __attribute__((unused)))
{
    return PAPI_OK;
}

int
open_xgmi_evt(void *arg)
{
    int papi_errno = PAPI_OK;
    ntv_event_t *event = (ntv_event_t *) arg;
    rsmi_status_t status;
    rsmi_event_group_t grp;

    switch (event->variant) {
        case ROCS_XGMI_VARIANT__MI50_0_NOP_TX:
        case ROCS_XGMI_VARIANT__MI50_0_REQUEST_TX:
        case ROCS_XGMI_VARIANT__MI50_0_RESPONSE_TX:
        case ROCS_XGMI_VARIANT__MI50_0_BEATS_TX:
        case ROCS_XGMI_VARIANT__MI50_1_NOP_TX:
        case ROCS_XGMI_VARIANT__MI50_1_REQUEST_TX:
        case ROCS_XGMI_VARIANT__MI50_1_RESPONSE_TX:
        case ROCS_XGMI_VARIANT__MI50_1_BEATS_TX:
            grp = RSMI_EVNT_GRP_XGMI;
            break;
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_0:
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_1:
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_2:
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_3:
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_4:
        case ROCS_XGMI_VARIANT__MI100_DATA_OUT_5:
            grp = RSMI_EVNT_GRP_XGMI_DATA_OUT;
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
            goto fn_fail;
    }

    uint32_t counters;
    status = rsmi_counter_available_counters_get_p(event->device, grp, &counters);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    if (counters < 1) {
        papi_errno = PAPI_ECNFLCT;
        goto fn_fail;
    }

    status = rsmi_dev_counter_create_p(event->device, event->variant, (rsmi_event_handle_t *) event->scratch);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
close_xgmi_evt(void *arg)
{
    int papi_errno = PAPI_OK;
    ntv_event_t *event = (ntv_event_t *) arg;
    rsmi_status_t status;

    status = rsmi_dev_counter_destroy_p(*(rsmi_event_handle_t *) event->scratch);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
    }

    return papi_errno;
}

int
start_xgmi_evt(void *arg)
{
    int papi_errno = PAPI_OK;
    ntv_event_t *event = (ntv_event_t *) arg;
    rsmi_status_t status;

    status = rsmi_counter_control_p(*(rsmi_event_handle_t *) event->scratch, RSMI_CNTR_CMD_START, NULL);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
    }

    return papi_errno;
}

int
stop_xgmi_evt(void *arg)
{
    int papi_errno = PAPI_OK;
    ntv_event_t *event = (ntv_event_t *) arg;
    rsmi_status_t status;

    status = rsmi_counter_control_p(*(rsmi_event_handle_t *) event->scratch, RSMI_CNTR_CMD_STOP, NULL);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
    }

    return papi_errno;
}

int
access_xgmi_evt(rocs_access_mode_e mode, void *arg)
{
    int papi_errno = PAPI_OK;
    ntv_event_t *event = (ntv_event_t *) arg;
    rsmi_status_t status;
    rsmi_counter_value_t value;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    status = rsmi_counter_read_p(*(rsmi_event_handle_t *) event->scratch, &value);
    if (status != RSMI_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
    }

    event->value = (int64_t) value.value;
    return papi_errno;
}

int
access_xgmi_bw(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t min, max;
    status = rsmi_minmax_bandwidth_get_p(event->device, (uint32_t) event->subvariant, &min, &max);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    switch (event->variant) {
        case ROCS_XGMI_BW_VARIANT__MIN:
            event->value = (int64_t) min;
            break;
        case ROCS_XGMI_BW_VARIANT__MAX:
            event->value = (int64_t) max;
            break;
        default:
            return PAPI_ENOSUPP;
    }

    return PAPI_OK;
}

int
access_rsmi_dev_count(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    event->value = (int64_t) device_count;
    return PAPI_OK;
}

int
access_rsmi_lib_version(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    rsmi_version_t version;
    status = rsmi_version_get_p(&version);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t)(version.major & 0x0000FFFF);
    event->value = (int64_t)(event->value << 16) | (version.minor & 0x0000FFFF);
    event->value = (int64_t)(event->value << 16) | (version.patch & 0x0000FFFF);
    return PAPI_OK;
}

int
access_rsmi_dev_driver_version_str(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_version_str_get_p(RSMI_SW_COMP_DRIVER, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint16_t data;
    status = rsmi_dev_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_subsystem_vendor_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint16_t data;
    status = rsmi_dev_subsystem_vendor_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_vendor_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint16_t data;
    status = rsmi_dev_vendor_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_unique_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_unique_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_subsystem_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint16_t data;
    status = rsmi_dev_subsystem_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_drm_render_minor(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint32_t data;
    status = rsmi_dev_drm_render_minor_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_overdrive_level(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (!(mode & event->mode)) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    uint32_t data;
    if (mode == ROCS_ACCESS_MODE__READ) {
        status = rsmi_dev_overdrive_level_get_p(event->device, &data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
        event->value = (int64_t) data;
    } else {
        data = (uint32_t) event->value;
        status = rsmi_dev_overdrive_level_set_p(event->device, data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }
    return PAPI_OK;
}

int
access_rsmi_dev_perf_level(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (!(mode & event->mode)) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    rsmi_dev_perf_level_t data;
    if (mode == ROCS_ACCESS_MODE__READ) {
        status = rsmi_dev_perf_level_get_p(event->device, &data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
        event->value = (int64_t) data;
    } else {
        data = (rsmi_dev_perf_level_t) event->value;
        status = rsmi_dev_perf_level_set_p(event->device, data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }
    return PAPI_OK;
}

int
access_rsmi_dev_memory_total(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_memory_total_get_p(event->device, event->variant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_memory_usage(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_memory_usage_get_p(event->device, event->variant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_memory_busy_percent(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint32_t data;
    status = rsmi_dev_memory_busy_percent_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_busy_percent(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint32_t data;
    status = rsmi_dev_busy_percent_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_pci_id(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_pci_id_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_pci_replay_counter(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_pci_replay_counter_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_pci_throughput(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data[3];
    status = rsmi_dev_pci_throughput_get_p(event->device, &data[0], &data[1], &data[2]);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    switch (event->variant) {
        case ROCS_PCI_THROUGHPUT_VARIANT__SENT:
            event->value = (int64_t) data[0];
            break;
        case ROCS_PCI_THROUGHPUT_VARIANT__RECEIVED:
            event->value = (int64_t) data[1];
            break;
        case ROCS_PCI_THROUGHPUT_VARIANT__MAX_PACKET_SIZE:
            event->value = (int64_t) data[2];
            break;
        default:
            return PAPI_EMISC;
    }

    return PAPI_OK;
}

int
access_rsmi_dev_power_profile_presets(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    rsmi_power_profile_status_t profile;
    status = rsmi_dev_power_profile_presets_get_p(event->device, event->variant, &profile);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    switch (event->variant) {
        case ROCS_POWER_PRESETS_VARIANT__COUNT:
            event->value = (int64_t) profile.num_profiles;
            break;
        case ROCS_POWER_PRESETS_VARIANT__AVAIL_PROFILES:
            event->value = (int64_t) profile.available_profiles;
            break;
        case ROCS_POWER_PRESETS_VARIANT__CURRENT:
            event->value = (int64_t) profile.current;
            break;
        default:
            return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_power_profile_set(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__WRITE || mode != event->mode) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    rsmi_power_profile_preset_masks_t mask = (rsmi_power_profile_preset_masks_t) event->value;
    status = rsmi_dev_power_profile_set_p(event->device, event->variant, mask);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_fan_reset(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__WRITE || mode != event->mode) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    status = rsmi_dev_fan_reset_p(event->device, event->subvariant);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_fan_rpms(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_fan_rpms_get_p(event->device, event->subvariant, &event->value);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_fan_speed_max(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_fan_speed_max_get_p(event->device, event->subvariant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_fan_speed(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    status = rsmi_dev_fan_speed_get_p(event->device, event->subvariant, &event->value);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_power_ave(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_power_avg_get_p(event->device, event->subvariant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_power_cap(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (!(mode & event->mode)) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    uint64_t data;
    if (mode == ROCS_ACCESS_MODE__READ) {
        status = rsmi_dev_power_cap_get_p(event->device, event->subvariant, &data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
        event->value = (int64_t) data;
    } else {
        data = (uint64_t) event->value;
        status = rsmi_dev_power_cap_set_p(event->device, event->subvariant, data);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
     }
    return PAPI_OK;
}

int
access_rsmi_dev_power_cap_range(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data[2];
    status = rsmi_dev_power_cap_range_get_p(event->device, event->subvariant, &data[0], &data[1]);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    switch (event->variant) {
        case ROCS_POWER_CAP_RANGE_VARIANT__MIN:
            event->value = (int64_t) data[1];
            break;
        case ROCS_POWER_CAP_RANGE_VARIANT__MAX:
            event->value = (int64_t) data[0];
            break;
        default:
            return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_temp_metric(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_temp_metric_get_p(event->device, event->subvariant, event->variant, &event->value);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_firmware_version(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_firmware_version_get_p(event->device, event->variant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_ecc_count(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    rsmi_error_count_t data;
    status = rsmi_dev_ecc_count_get_p(event->device, event->variant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    switch (event->subvariant) {
        case ROCS_ECC_COUNT_SUBVARIANT__CORRECTABLE:
            event->value = (int64_t) data.correctable_err;
            break;
        case ROCS_ECC_COUNT_SUBVARIANT__UNCORRECTABLE:
            event->value = (int64_t) data.uncorrectable_err;
            break;
        default:
            return PAPI_EMISC;
    }
    return PAPI_OK;
}

int
access_rsmi_dev_ecc_enabled(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    uint64_t data;
    status = rsmi_dev_ecc_enabled_get_p(event->device, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_ecc_status(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    rsmi_ras_err_state_t data;
    status = rsmi_dev_ecc_status_get_p(event->device, event->variant, &data);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->value = (int64_t) data;
    return PAPI_OK;
}

int
access_rsmi_dev_gpu_clk_freq(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (!(mode & event->mode)) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    if (mode == ROCS_ACCESS_MODE__READ) {
        int table_id = ROCS_GPU_CLK_FREQ_VARIANT__NUM * event->device + event->variant;
        status = rsmi_dev_gpu_clk_freq_get_p(event->device, event->variant, &freq_table[table_id]);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
        uint32_t current, freq_id;
        switch (event->subvariant) {
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__COUNT:
                event->value = (int64_t) freq_table[table_id].num_supported;
                break;
            case ROCS_GPU_CLK_FREQ_SUBVARIANT__CURRENT:
                current = freq_table[table_id].current;
                event->value = (int64_t) freq_table[table_id].frequency[current];
                break;
            default:
                freq_id = event->subvariant - ROCS_GPU_CLK_FREQ_SUBVARIANT__NUM;
                event->value = (int64_t) freq_table[table_id].frequency[freq_id];
        }
    } else {
        uint64_t data = (uint64_t) event->value;
        uint32_t freq_id = (uint32_t) (event->device * ROCS_GPU_CLK_FREQ_VARIANT__NUM + event->variant);
        uint64_t mask = (1 << freq_table[freq_id].num_supported) - 1;
        if ((data & mask) == 0) {
            return PAPI_EINVAL;
        }
        status = rsmi_dev_gpu_clk_freq_set_p(event->device, event->variant, data & mask);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }
    return PAPI_OK;
}

int
access_rsmi_dev_pci_bandwidth(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (!(mode & event->mode)) {
        /* Return error code as counter value to distinguish
         * this case from a successful read */
        event->value = PAPI_ENOSUPP;
        return PAPI_OK;
    }

    rsmi_status_t status;
    if (mode == ROCS_ACCESS_MODE__READ) {
        uint32_t current;
        status = rsmi_dev_pci_bandwidth_get_p(event->device, &pcie_table[event->device]);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }

        switch (event->variant) {
            case ROCS_PCI_BW_VARIANT__COUNT:
                event->value = (int64_t) pcie_table[event->device].transfer_rate.num_supported;
                break;
            case ROCS_PCI_BW_VARIANT__CURRENT:
                current = pcie_table[event->device].transfer_rate.current;
                event->value = (int64_t) pcie_table[event->device].transfer_rate.frequency[current];
                break;
            case ROCS_PCI_BW_VARIANT__RATE_IDX:
            case ROCS_PCI_BW_VARIANT__LANE_IDX:
                current = event->subvariant;
                event->value = (int64_t) pcie_table[event->device].transfer_rate.frequency[current];
                break;
            default:
                return PAPI_EMISC;
        }
    } else {
        uint64_t data = (uint64_t) event->value;
        uint64_t mask = (1 << pcie_table[event->device].transfer_rate.num_supported) - 1;
        if ((data & mask) == 0) {
            return PAPI_EINVAL;
        }
        status = rsmi_dev_pci_bandwidth_set_p(event->device, data & mask);
        if (status != RSMI_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }
    return PAPI_OK;
}

int
access_rsmi_dev_brand(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_brand_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_name(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_name_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_serial_number(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_serial_number_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_subsystem_name(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_subsystem_name_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_vbios_version(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_vbios_version_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}

int
access_rsmi_dev_vendor_name(rocs_access_mode_e mode, void *arg)
{
    ntv_event_t *event = (ntv_event_t *) arg;

    if (mode != ROCS_ACCESS_MODE__READ || mode != event->mode) {
        return PAPI_ENOSUPP;
    }

    rsmi_status_t status;
    status = rsmi_dev_vendor_name_get_p(event->device, event->scratch, PAPI_MAX_STR_LEN - 1);
    if (status != RSMI_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    event->scratch[PAPI_MAX_STR_LEN - 1] = 0;
    event->value = (int64_t) event->scratch;
    return PAPI_OK;
}
