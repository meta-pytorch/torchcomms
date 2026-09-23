/**
 * @file    amds_funcs.h
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#ifndef AMDS_FUNCS_H
#define AMDS_FUNCS_H

#define AMD_SMI_GPU_FUNCTIONS_BASE(_)                                          \
  _(amdsmi_init_p, amdsmi_status_t, (uint64_t))                                \
  _(amdsmi_shut_down_p, amdsmi_status_t, (void))                               \
  _(amdsmi_get_socket_handles_p, amdsmi_status_t,                              \
    (uint32_t *, amdsmi_socket_handle *))                                      \
  _(amdsmi_get_processor_handles_by_type_p, amdsmi_status_t,                   \
    (amdsmi_socket_handle, processor_type_t, amdsmi_processor_handle *,        \
     uint32_t *))                                                              \
  _(amdsmi_get_processor_handles_p, amdsmi_status_t,                           \
    (amdsmi_socket_handle, uint32_t *, amdsmi_processor_handle *))             \
  _(amdsmi_get_processor_info_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, size_t, char *))                                 \
  _(amdsmi_get_processor_type_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, processor_type_t *))                             \
  _(amdsmi_get_socket_info_p, amdsmi_status_t,                                 \
    (amdsmi_socket_handle, size_t, char *))                                    \
  _(amdsmi_get_utilization_count_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, amdsmi_utilization_counter_t *, uint32_t,        \
     uint64_t *))                                                              \
  _(amdsmi_get_violation_status_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, amdsmi_violation_status_t *))                    \
  _(amdsmi_get_temp_metric_p, amdsmi_status_t,                                 \
    (amdsmi_processor_handle, amdsmi_temperature_type_t,                       \
     amdsmi_temperature_metric_t, int64_t *))                                  \
  _(amdsmi_get_gpu_fan_rpms_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, uint32_t, int64_t *))                            \
  _(amdsmi_get_gpu_fan_speed_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, uint32_t, int64_t *))                            \
  _(amdsmi_get_gpu_fan_speed_max_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, uint32_t, int64_t *))                            \
  _(amdsmi_set_gpu_fan_speed_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, uint32_t, uint64_t))                             \
  _(amdsmi_get_total_memory_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *))               \
  _(amdsmi_get_memory_usage_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t *))               \
  _(amdsmi_get_gpu_activity_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, amdsmi_engine_usage_t *))                        \
  _(amdsmi_get_power_cap_info_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, uint32_t, amdsmi_power_cap_info_t *))            \
  _(amdsmi_get_gpu_power_cap_set_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, uint32_t, uint64_t))                             \
  _(amdsmi_get_power_info_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, amdsmi_power_info_t *))                          \
  _(amdsmi_set_power_cap_p, amdsmi_status_t,                                   \
    (amdsmi_processor_handle, uint32_t, uint64_t))                             \
  _(amdsmi_get_gpu_pci_throughput_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, uint64_t *, uint64_t *, uint64_t *))             \
  _(amdsmi_get_gpu_pci_replay_counter_p, amdsmi_status_t,                      \
    (amdsmi_processor_handle, uint64_t *))                                     \
  _(amdsmi_get_clk_freq_p, amdsmi_status_t,                                    \
    (amdsmi_processor_handle, amdsmi_clk_type_t, amdsmi_frequencies_t *))      \
  _(amdsmi_get_clock_info_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, amdsmi_clk_type_t, amdsmi_clk_info_t *))         \
  _(amdsmi_set_clk_freq_p, amdsmi_status_t,                                    \
    (amdsmi_processor_handle, amdsmi_clk_type_t, uint64_t))                    \
  _(amdsmi_get_gpu_metrics_info_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, amdsmi_gpu_metrics_t *))                         \
  _(amdsmi_get_lib_version_p, amdsmi_status_t, (amdsmi_version_t *))           \
  _(amdsmi_get_gpu_driver_info_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, amdsmi_driver_info_t *))                         \
  _(amdsmi_get_gpu_asic_info_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, amdsmi_asic_info_t *))                           \
  _(amdsmi_get_gpu_board_info_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_board_info_t *))                          \
  _(amdsmi_get_fw_info_p, amdsmi_status_t,                                     \
    (amdsmi_processor_handle, amdsmi_fw_info_t *))                             \
  _(amdsmi_get_gpu_vbios_info_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_vbios_info_t *))                          \
  _(amdsmi_get_gpu_device_uuid_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, unsigned int *, char *))                         \
  _(amdsmi_get_gpu_vendor_name_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, char *, size_t))                                 \
  _(amdsmi_get_gpu_vram_vendor_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, char *, uint32_t))                               \
  _(amdsmi_get_gpu_subsystem_name_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, char *, size_t))                                 \
  _(amdsmi_get_link_metrics_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, amdsmi_link_metrics_t *))                        \
  _(amdsmi_get_minmax_bandwidth_between_processors_p, amdsmi_status_t,        \
    (amdsmi_processor_handle, amdsmi_processor_handle, uint64_t *,            \
     uint64_t *))                                                             \
  _(amdsmi_get_gpu_process_list_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, uint32_t *, amdsmi_proc_info_t *))               \
  _(amdsmi_get_gpu_ecc_enabled_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, uint64_t *))                                     \
  _(amdsmi_get_gpu_total_ecc_count_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, amdsmi_error_count_t *))                         \
  _(amdsmi_get_gpu_ecc_count_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, amdsmi_gpu_block_t, amdsmi_error_count_t *))     \
  _(amdsmi_get_gpu_ecc_status_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_gpu_block_t, amdsmi_ras_err_state_t *))   \
  _(amdsmi_get_gpu_compute_partition_p, amdsmi_status_t,                       \
    (amdsmi_processor_handle, char *, uint32_t))                               \
  _(amdsmi_get_gpu_memory_partition_p, amdsmi_status_t,                        \
    (amdsmi_processor_handle, char *, uint32_t))                               \
  _(amdsmi_get_gpu_accelerator_partition_profile_p, amdsmi_status_t,           \
    (amdsmi_processor_handle, amdsmi_accelerator_partition_profile_t *,        \
     uint32_t *))                                                              \
  _(amdsmi_get_gpu_id_p, amdsmi_status_t,                                      \
    (amdsmi_processor_handle, uint16_t *))                                     \
  _(amdsmi_get_gpu_revision_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, uint16_t *))                                     \
  _(amdsmi_get_gpu_subsystem_id_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, uint16_t *))                                     \
  _(amdsmi_get_gpu_process_isolation_p, amdsmi_status_t,                       \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_gpu_xcd_counter_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, uint16_t *))                                     \
  _(amdsmi_get_gpu_pci_bandwidth_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, amdsmi_pcie_bandwidth_t *))                      \
  _(amdsmi_get_gpu_bdf_id_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, uint64_t *))                                     \
  _(amdsmi_get_gpu_device_bdf_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_bdf_t *))                                 \
  _(amdsmi_get_gpu_topo_numa_affinity_p, amdsmi_status_t,                      \
    (amdsmi_processor_handle, int32_t *))                                      \
  _(amdsmi_topo_get_numa_node_number_p, amdsmi_status_t,                       \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_topo_get_link_weight_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, amdsmi_processor_handle, uint64_t *))            \
  _(amdsmi_topo_get_link_type_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_processor_handle, uint64_t *,             \
     amdsmi_link_type_compat_t *))                                            \
  _(amdsmi_topo_get_p2p_status_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, amdsmi_processor_handle, amdsmi_link_type_compat_t *,\
     amdsmi_p2p_capability_t *))                                              \
  _(amdsmi_is_P2P_accessible_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, amdsmi_processor_handle, bool *))                \
  _(amdsmi_get_link_topology_nearest_p, amdsmi_status_t,                       \
    (amdsmi_processor_handle, amdsmi_link_type_t,                              \
     amdsmi_topology_nearest_t *))                                            \
  _(amdsmi_get_energy_count_p, amdsmi_status_t,                                \
    (amdsmi_processor_handle, uint64_t *, float *, uint64_t *))                \
  _(amdsmi_get_gpu_power_profile_presets_p, amdsmi_status_t,                   \
    (amdsmi_processor_handle, uint32_t, amdsmi_power_profile_status_t *))      \
  _(amdsmi_get_gpu_cache_info_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_gpu_cache_info_t *))                      \
  _(amdsmi_get_gpu_mem_overdrive_level_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_gpu_od_volt_curve_regions_p, amdsmi_status_t,                   \
    (amdsmi_processor_handle, uint32_t *, amdsmi_freq_volt_region_t *))        \
  _(amdsmi_get_gpu_od_volt_info_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, amdsmi_od_volt_freq_data_t *))                   \
  _(amdsmi_get_gpu_overdrive_level_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_gpu_perf_level_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_dev_perf_level_t *))                      \
  _(amdsmi_get_gpu_pm_metrics_info_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, amdsmi_name_value_t **, uint32_t *))             \
  _(amdsmi_get_gpu_ras_feature_info_p, amdsmi_status_t,                        \
    (amdsmi_processor_handle, amdsmi_ras_feature_t *))                         \
  _(amdsmi_get_gpu_ras_block_features_enabled_p, amdsmi_status_t,              \
    (amdsmi_processor_handle, amdsmi_gpu_block_t, amdsmi_ras_err_state_t *))   \
  _(amdsmi_get_gpu_reg_table_info_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, amdsmi_reg_type_t, amdsmi_name_value_t **,       \
     uint32_t *))                                                              \
  _(amdsmi_get_gpu_volt_metric_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, amdsmi_voltage_type_t, amdsmi_voltage_metric_t,  \
     int64_t *))                                                               \
  _(amdsmi_get_gpu_vram_info_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, amdsmi_vram_info_t *))                           \
  _(amdsmi_get_gpu_vram_usage_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_vram_usage_t *))                          \
  _(amdsmi_get_pcie_info_p, amdsmi_status_t,                                   \
    (amdsmi_processor_handle, amdsmi_pcie_info_t *))                           \
  _(amdsmi_get_processor_count_from_handles_p, amdsmi_status_t,                \
    (amdsmi_processor_handle *, uint32_t *, uint32_t *, uint32_t *,            \
     uint32_t *))                                                              \
  _(amdsmi_get_soc_pstate_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, amdsmi_dpm_policy_t *))                          \
  _(amdsmi_get_xgmi_plpd_p, amdsmi_status_t,                                   \
    (amdsmi_processor_handle, amdsmi_dpm_policy_t *))                          \
  _(amdsmi_get_gpu_bad_page_info_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, uint32_t *, amdsmi_retired_page_record_t *))     \
  _(amdsmi_get_gpu_bad_page_threshold_p, amdsmi_status_t,                      \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_power_info_v2_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, uint32_t, amdsmi_power_info_t *))                \
  _(amdsmi_init_gpu_event_notification_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle))                                                 \
  _(amdsmi_set_gpu_event_notification_mask_p, amdsmi_status_t,                 \
    (amdsmi_processor_handle, uint64_t))                                       \
  _(amdsmi_get_gpu_event_notification_p, amdsmi_status_t,                      \
    (int, uint32_t *, amdsmi_evt_notification_data_t *))                       \
  _(amdsmi_stop_gpu_event_notification_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle))                                               \
  _(amdsmi_gpu_counter_group_supported_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle, amdsmi_event_group_t))                          \
  _(amdsmi_get_gpu_available_counters_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle, amdsmi_event_group_t, uint32_t *))              \
  _(amdsmi_gpu_create_counter_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, amdsmi_event_type_t,                            \
     amdsmi_event_handle_t *))                                               \
  _(amdsmi_gpu_control_counter_p, amdsmi_status_t,                             \
    (amdsmi_event_handle_t, amdsmi_counter_command_t, void *))                \
  _(amdsmi_gpu_read_counter_p, amdsmi_status_t,                                \
    (amdsmi_event_handle_t, amdsmi_counter_value_t *))                        \
  _(amdsmi_get_gpu_kfd_info_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, amdsmi_kfd_info_t *))                           \
  _(amdsmi_is_gpu_memory_partition_supported_p, amdsmi_status_t,              \
    (amdsmi_processor_handle, bool *))                                        \
  _(amdsmi_get_gpu_memory_reserved_pages_p, amdsmi_status_t,                  \
    (amdsmi_processor_handle, uint32_t *, amdsmi_retired_page_record_t *))    \
  _(amdsmi_get_gpu_metrics_header_info_p, amdsmi_status_t,                    \
    (amdsmi_processor_handle, amd_metrics_table_header_t *))                  \
  _(amdsmi_get_xgmi_info_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, amdsmi_xgmi_info_t *))                          \
  _(amdsmi_gpu_xgmi_error_status_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, amdsmi_xgmi_status_t *))                        \
  _(amdsmi_is_gpu_power_management_enabled_p, amdsmi_status_t,                \
    (amdsmi_processor_handle, bool *))                                        \
  _(amdsmi_gpu_validate_ras_eeprom_p, amdsmi_status_t,                        \
    (amdsmi_processor_handle))                                               \
  _(amdsmi_gpu_destroy_counter_p, amdsmi_status_t,                             \
    (amdsmi_event_handle_t))

#if AMDSMI_LIB_VERSION_MAJOR >= 25
#define AMD_SMI_GPU_FUNCTIONS(_) \
  AMD_SMI_GPU_FUNCTIONS_BASE(_) \
  _(amdsmi_get_gpu_memory_partition_config_p, amdsmi_status_t, \
    (amdsmi_processor_handle, amdsmi_memory_partition_config_t *)) \
  _(amdsmi_get_gpu_xgmi_link_status_p, amdsmi_status_t, \
    (amdsmi_processor_handle, amdsmi_xgmi_link_status_t *)) \
  _(amdsmi_get_gpu_enumeration_info_p, amdsmi_status_t, \
    (amdsmi_processor_handle, amdsmi_enumeration_info_t *)) \
  _(amdsmi_get_gpu_virtualization_mode_p, amdsmi_status_t, \
    (amdsmi_processor_handle, amdsmi_virtualization_mode_t *))
#else
#define AMD_SMI_GPU_FUNCTIONS(_) AMD_SMI_GPU_FUNCTIONS_BASE(_)
#endif

#define AMD_SMI_CPU_FUNCTIONS(_)                                               \
  _(amdsmi_get_cpu_handles_p, amdsmi_status_t,                                 \
    (uint32_t *, amdsmi_processor_handle *))                                   \
  _(amdsmi_get_cpucore_handles_p, amdsmi_status_t,                             \
    (uint32_t *, amdsmi_processor_handle *))                                   \
  _(amdsmi_get_cpu_socket_power_p, amdsmi_status_t,                            \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_socket_power_cap_p, amdsmi_status_t,                        \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_socket_power_cap_max_p, amdsmi_status_t,                    \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_core_energy_p, amdsmi_status_t,                             \
    (amdsmi_processor_handle, uint64_t *))                                     \
  _(amdsmi_get_cpu_socket_energy_p, amdsmi_status_t,                           \
    (amdsmi_processor_handle, uint64_t *))                                     \
  _(amdsmi_get_cpu_smu_fw_version_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, amdsmi_smu_fw_version_t *))                      \
  _(amdsmi_get_threads_per_core_p, amdsmi_status_t, (uint32_t *))              \
  _(amdsmi_get_cpu_family_p, amdsmi_status_t, (uint32_t *))                    \
  _(amdsmi_get_cpu_model_p, amdsmi_status_t, (uint32_t *))                     \
  _(amdsmi_get_cpu_core_boostlimit_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_socket_current_active_freq_limit_p, amdsmi_status_t,        \
    (amdsmi_processor_handle, uint16_t *, char **))                            \
  _(amdsmi_get_cpu_socket_freq_range_p, amdsmi_status_t,                       \
    (amdsmi_processor_handle, uint16_t *, uint16_t *))                         \
  _(amdsmi_get_cpu_core_current_freq_limit_p, amdsmi_status_t,                 \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_cclk_limit_p, amdsmi_status_t,                              \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_current_io_bandwidth_p, amdsmi_status_t,                    \
    (amdsmi_processor_handle, amdsmi_link_id_bw_type_t, uint32_t *))           \
  _(amdsmi_get_cpu_current_xgmi_bw_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, amdsmi_link_id_bw_type_t, uint32_t *))           \
  _(amdsmi_get_cpu_ddr_bw_p, amdsmi_status_t,                                  \
    (amdsmi_processor_handle, amdsmi_ddr_bw_metrics_t *))                      \
  _(amdsmi_get_cpu_fclk_mclk_p, amdsmi_status_t,                               \
    (amdsmi_processor_handle, uint32_t *, uint32_t *))                         \
  _(amdsmi_get_cpu_hsmp_driver_version_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle, amdsmi_hsmp_driver_version_t *))                 \
  _(amdsmi_get_cpu_hsmp_proto_ver_p, amdsmi_status_t,                          \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_prochot_status_p, amdsmi_status_t,                         \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_pwr_svi_telemetry_all_rails_p, amdsmi_status_t,             \
    (amdsmi_processor_handle, uint32_t *))                                     \
  _(amdsmi_get_cpu_dimm_temp_range_and_refresh_rate_p, amdsmi_status_t,        \
    (amdsmi_processor_handle, uint8_t, amdsmi_temp_range_refresh_rate_t *))    \
  _(amdsmi_get_cpu_dimm_power_consumption_p, amdsmi_status_t,                  \
    (amdsmi_processor_handle, uint8_t, amdsmi_dimm_power_t *))                 \
  _(amdsmi_get_cpu_dimm_thermal_sensor_p, amdsmi_status_t,                     \
    (amdsmi_processor_handle, uint8_t, amdsmi_dimm_thermal_t *))

#endif /* AMDS_FUNCS_H */
