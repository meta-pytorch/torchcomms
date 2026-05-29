/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    linux-sensors-ppc.h
 * CVS:     $Id$
 *
 * @author  PAPI team UTK/ICL
 *          dgenet@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief OCC Inband Sensors component for PowerPC
 *  This file contains the source code for a component that enables
 *  PAPI to read counters and sensors on PowerPC (Power9) architecture.
 */

#ifndef _sensors_ppc_H
#define _sensors_ppc_H

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#define papi_sensors_ppc_lock() _papi_hwi_lock(COMPONENT_LOCK);
#define papi_sensors_ppc_unlock() _papi_hwi_unlock(COMPONENT_LOCK);

typedef struct _sensors_ppc_register {
    unsigned int selector;
} _sensors_ppc_register_t;

typedef struct _sensors_ppc_native_event_entry {
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  int socket_id;
  int component_id;
  int event_id;
  int type;
  int return_type;
  _sensors_ppc_register_t resources;
} _sensors_ppc_native_event_entry_t;

typedef struct _sensors_ppc_reg_alloc {
    _sensors_ppc_register_t ra_bits;
} _sensors_ppc_reg_alloc_t;

static int num_events=0;

typedef enum occ_sensor_type_e {
    OCC_SENSOR_TYPE_GENERIC      = 0x0001,
    OCC_SENSOR_TYPE_CURRENT      = 0x0002,
    OCC_SENSOR_TYPE_VOLTAGE      = 0x0004,
    OCC_SENSOR_TYPE_TEMPERATURE  = 0x0008,
    OCC_SENSOR_TYPE_UTILIZATION  = 0x0010,
    OCC_SENSOR_TYPE_TIME         = 0x0020,
    OCC_SENSOR_TYPE_FREQUENCY    = 0x0040,
    OCC_SENSOR_TYPE_POWER        = 0x0080,
    OCC_SENSOR_TYPE_PERFORMANCE  = 0x0200,
} occ_sensor_type_t;

typedef enum occ_sensor_loc_e {
    OCC_SENSOR_LOC_SYSTEM        = 0x0001,
    OCC_SENSOR_LOC_PROCESSOR     = 0x0002,
    OCC_SENSOR_LOC_PARTITION     = 0x0004,
    OCC_SENSOR_LOC_MEMORY        = 0x0008,
    OCC_SENSOR_LOC_VRM           = 0x0010,
    OCC_SENSOR_LOC_OCC           = 0x0020,
    OCC_SENSOR_LOC_CORE          = 0x0040,
    OCC_SENSOR_LOC_GPU           = 0x0080,
    OCC_SENSOR_LOC_QUAD          = 0x0100,
} occ_sensor_loc_t;

#define OCC_SENSOR_READING_FULL      0x01
#define OCC_SENSOR_READING_COUNTER   0x02

static char *pkg_sys_name = "occ_inband_sensors";
static mode_t   pkg_sys_flag = O_RDONLY;

/* 8 OCCs, starting at OCC_SENSOR_DATA_BLOCK_OFFSET
 * OCC0: 0x00580000 -> 0x005A57FF
 * OCC1: 0x005A5800 -> 0x005CAFFF
 * Each zone is 150kB (OCC_SENSOR_DATA_BLOCK_SIZE)
 * OCC7: 0x00686800 -> 0x006ABFFF*/

#define MAX_OCCS                     8
#define OCC_SENSOR_DATA_BLOCK_OFFSET 0x00580000
#define OCC_SENSOR_DATA_BLOCK_SIZE   0x00025800
#define OCC_PING_DATA_BLOCK_SIZE     0xA000
#define OCC_REFRESH_TIME             100000

/* In the 150kB, map the beginning to */
typedef struct occ_sensor_data_header_s {
    uint8_t valid; /* 0x01 means the block can be read */
    uint8_t version;
    uint16_t nr_sensors; /* number of sensors! */
    uint8_t reading_version; /* ping pong version */
    uint8_t pad[3];
    uint32_t names_offset;
    uint8_t names_version;
    uint8_t name_length;
    uint16_t reserved;
    uint32_t reading_ping_offset;
    uint32_t reading_pong_offset;
} __attribute__((__packed__)) occ_sensor_data_header_t;
/* That header is reset after each reboot */

struct occ_sensor_data_header_s *occ_hdr[MAX_OCCS];
static int event_fd;
static long long last_refresh[MAX_OCCS];
static int num_occs;
static int occ_num_events[MAX_OCCS+1];
static uint32_t *ping[MAX_OCCS], *pong[MAX_OCCS];
static uint32_t *double_ping[MAX_OCCS], *double_pong[MAX_OCCS];

#define MAX_CHARS_SENSOR_NAME        16
#define MAX_CHARS_SENSOR_UNIT        4

/* After 1kB, the list of sensor names, units */
/* map an array of size header->nr_sensors */
/* the following struct, */
typedef struct occ_sensor_name_s {
    char name[MAX_CHARS_SENSOR_NAME];
    char units[MAX_CHARS_SENSOR_UNIT];
    uint16_t gsid;
    uint32_t freq;
    uint32_t scale_factor;
    uint16_t type;
    uint16_t location;
    uint8_t structure_type; /* determine size+format of sensor */
    uint32_t reading_offset;
    uint8_t sensor_data;
    uint8_t pad[8];
} __attribute__((__packed__)) occ_sensor_name_t;

struct occ_sensor_name_s *occ_names[MAX_OCCS];

/* The following 4kB, size of a page, has to be skipped */

/* Following 40kB is the ping buffer */
/* Followed by another 4kB of skippable memory */
/* Finally, 40kB for the pong buffer */

typedef struct occ_sensor_record_s {
	uint16_t gsid;
	uint64_t timestamp;
	uint16_t sample; /* latest value */
	uint16_t sample_min; /*min max since reboot */
	uint16_t sample_max;
	uint16_t csm_min;/* since CSM reset */
	uint16_t csm_max;
	uint16_t profiler_min; /* since prof reset */
	uint16_t profiler_max;
	uint16_t job_scheduler_min; /* since job sched reset */
	uint16_t job_scheduler_max;
	uint64_t accumulator; /* accu if it makes sense */
	uint32_t update_tag; /* tics since between that value and previous one */
	uint8_t pad[8];
} __attribute__((__packed__)) occ_sensor_record_t;

typedef struct occ_sensor_counter_s {
	uint16_t gsid;
	uint64_t timestamp;
	uint64_t accumulator;
	uint8_t sample;
	uint8_t pad[5];
} __attribute__((__packed__)) occ_sensor_counter_t;

typedef enum occ_sensors_mask_e {
  OCC_SENSORS_SAMPLE              = 0,
  OCC_SENSORS_SAMPLE_MIN          = 1,
  OCC_SENSORS_SAMPLE_MAX          = 2,
  OCC_SENSORS_CSM_MIN             = 3,
  OCC_SENSORS_CSM_MAX             = 4,
  OCC_SENSORS_PROFILER_MIN        = 5,
  OCC_SENSORS_PROFILER_MAX        = 6,
  OCC_SENSORS_JOB_SCHED_MIN       = 7,
  OCC_SENSORS_JOB_SCHED_MAX       = 8,
  OCC_SENSORS_ACCUMULATOR_TAG     = 9,
  OCC_SENSORS_MASKS
} occ_sensors_mask_t;

static const char* sensors_ppc_fake_qualifiers[] = {"", ":min", ":max", ":csm_min",
  ":csm_max", ":profiler_min", ":profiler_max", ":job_scheduler_min", ":job_scheduler_max", ":accumulator", NULL};
static const char *sensors_ppc_fake_qualif_desc[] = {
  "Last sample of this sensor",
  "Minimum value since last OCC reset (node reboot)",
  "Maximum value since last OCC reset (node reboot)",
  "Minimum value since last reset request by CSM",
  "Maximum value since last reset request by CSM",
  "Minimum value since last reset request by profiler",
  "Maximum value since last reset request by profiler",
  "Minimum value since last reset by job scheduler",
  "Maximum value since last reset by job scheduler",
  "Accumulator register for this sensor", NULL};

#define SENSORS_PPC_MAX_COUNTERS MAX_OCCS * 512 * OCC_SENSORS_MASKS

typedef struct _sensors_ppc_control_state {
  long long count[SENSORS_PPC_MAX_COUNTERS];
  long long which_counter[SENSORS_PPC_MAX_COUNTERS];
  long long need_difference[SENSORS_PPC_MAX_COUNTERS];
  uint32_t occ, scale;
} _sensors_ppc_control_state_t;

typedef struct _sensors_ppc_context {
  long long start_value[SENSORS_PPC_MAX_COUNTERS];
  _sensors_ppc_control_state_t state;
} _sensors_ppc_context_t;

#endif /* _sensors_ppc_H */
