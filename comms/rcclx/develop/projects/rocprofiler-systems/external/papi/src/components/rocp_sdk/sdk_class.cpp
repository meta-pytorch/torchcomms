/**
 * @file    sdk_class.cpp
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 */

#include "sdk_class.hpp"

namespace papi_rocpsdk
{
using agent_map_t = std::map<uint64_t, const rocprofiler_agent_v0_t*>;
using dim_t = std::pair<uint64_t, unsigned long>;
using dim_vector_t = std::vector< dim_t >;

static inline bool dimensions_match( dim_vector_t dim_instances, dim_vector_t recorded_dims );

typedef struct {
    rocprofiler_counter_info_v0_t counter_info;
    std::vector<rocprofiler_record_dimension_info_t> dim_info;
} base_event_info_t;

typedef struct {
    uint64_t qualifiers_present;
    std::string event_inst_name;
    rocprofiler_counter_info_v0_t counter_info;
    std::vector<rocprofiler_record_dimension_info_t> dim_info;
    dim_vector_t dim_instances;
    int device;
} event_instance_info_t;

typedef struct {
    rocprofiler_counter_id_t counter_id;
    uint64_t device;
    dim_vector_t recorded_dims;
} rec_info_t;

std::atomic<unsigned int> _global_papi_event_count{0};
std::atomic<unsigned int> _base_event_count{0};
#if (__cplusplus >= 201703L) // c++17
static std::shared_mutex profile_cache_mutex = {};
#define SHARED_LOCK std::shared_lock
#define UNIQUE_LOCK std::unique_lock
#elif (__cplusplus >= 201402L) // c++14
static std::shared_timed_mutex profile_cache_mutex = {};
#define SHARED_LOCK std::shared_lock<std::shared_timed_mutex>
#define UNIQUE_LOCK std::unique_lock<std::shared_timed_mutex>
#elif (__cplusplus >= 201103L) // c++11
static std::mutex profile_cache_mutex = {};
#define SHARED_LOCK std::lock_guard<std::mutex>
#define UNIQUE_LOCK std::lock_guard<std::mutex>
#else
#error "c++11 or higher is required"
#endif
static std::mutex agent_mutex = {};
static std::condition_variable agent_cond_var = {};
static bool data_is_ready = false;
static std::string _rocp_sdk_error_string;
static long long int *_counter_values = NULL;
static int rpsdk_profiling_mode = RPSDK_MODE_DISPATCH;

static agent_map_t gpu_agents = agent_map_t{};

static std::unordered_map<std::string, base_event_info_t>  base_events_by_name = {};

static std::set<int> active_device_set = {};
static vendorp_ctx_t active_event_set_ctx = NULL;
static std::vector<bool> index_mapping;

static std::unordered_map<uint64_t, rocprofiler_profile_config_id_t> rpsdk_profile_cache = {};
static std::unordered_map<unsigned int, event_instance_info_t> papi_id_to_event_instance = {};
static std::unordered_map<std::string, unsigned int> event_instance_name_to_papi_id = {};

/* *** */
typedef rocprofiler_status_t (* rocprofiler_flush_buffer_t) (rocprofiler_buffer_id_t buffer_id);

typedef rocprofiler_status_t (* rocprofiler_sample_device_counting_service_t) (rocprofiler_context_id_t context_id, rocprofiler_user_data_t user_data, rocprofiler_counter_flag_t flags, rocprofiler_record_counter_t* output_records, size_t* rec_count);

typedef rocprofiler_status_t (* rocprofiler_configure_callback_dispatch_counting_service_t) (rocprofiler_context_id_t context_id, rocprofiler_dispatch_counting_service_callback_t dispatch_callback, void *dispatch_callback_args, rocprofiler_profile_counting_record_callback_t record_callback, void *record_callback_args);

typedef rocprofiler_status_t (* rocprofiler_configure_device_counting_service_t) (rocprofiler_context_id_t context_id, rocprofiler_buffer_id_t buffer_id, rocprofiler_agent_id_t agent_id, rocprofiler_device_counting_service_callback_t cb, void *user_data);



typedef rocprofiler_status_t (* rocprofiler_create_buffer_t) (rocprofiler_context_id_t context, unsigned long size, unsigned long watermark, rocprofiler_buffer_policy_t policy, rocprofiler_buffer_tracing_cb_t callback, void *callback_data, rocprofiler_buffer_id_t *buffer_id);

typedef rocprofiler_status_t (* rocprofiler_create_context_t) (rocprofiler_context_id_t *context_id);

typedef rocprofiler_status_t (* rocprofiler_start_context_t) (rocprofiler_context_id_t context_id);

typedef rocprofiler_status_t (* rocprofiler_stop_context_t) (rocprofiler_context_id_t context_id);

typedef rocprofiler_status_t (* rocprofiler_context_is_valid_t) (rocprofiler_context_id_t context_id, int *status);

typedef rocprofiler_status_t (* rocprofiler_context_is_active_t) (rocprofiler_context_id_t context_id, int *status);

typedef rocprofiler_status_t (* rocprofiler_create_profile_config_t) (rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t *counters_list, unsigned long counters_count, rocprofiler_profile_config_id_t *config_id);

typedef rocprofiler_status_t (* rocprofiler_destroy_profile_config_t) (rocprofiler_profile_config_id_t config_id);

typedef rocprofiler_status_t (* rocprofiler_force_configure_t) (rocprofiler_configure_func_t configure_func);

typedef const char *         (* rocprofiler_get_status_string_t) (rocprofiler_status_t status);

typedef rocprofiler_status_t (* rocprofiler_get_thread_id_t) (rocprofiler_thread_id_t *tid);

typedef rocprofiler_status_t (* rocprofiler_is_finalized_t) (int *status);

typedef rocprofiler_status_t (* rocprofiler_is_initialized_t) (int *status);

typedef rocprofiler_status_t (* rocprofiler_iterate_agent_supported_counters_t) (rocprofiler_agent_id_t agent_id, rocprofiler_available_counters_cb_t cb, void* user_data);

typedef rocprofiler_status_t (* rocprofiler_iterate_counter_dimensions_t) (rocprofiler_counter_id_t id, rocprofiler_available_dimensions_cb_t info_cb, void *user_data);

typedef rocprofiler_status_t (* rocprofiler_query_available_agents_t) (rocprofiler_agent_version_t version, rocprofiler_query_available_agents_cb_t callback, unsigned long agent_size, void *user_data);

typedef rocprofiler_status_t (* rocprofiler_query_counter_info_t) (rocprofiler_counter_id_t counter_id, rocprofiler_counter_info_version_id_t version, void *info);

typedef rocprofiler_status_t (* rocprofiler_query_counter_instance_count_t) (rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t counter_id, unsigned long *instance_count);

typedef rocprofiler_status_t (* rocprofiler_query_record_counter_id_t) (rocprofiler_counter_instance_id_t id, rocprofiler_counter_id_t *counter_id);

typedef rocprofiler_status_t (* rocprofiler_query_record_dimension_position_t) (rocprofiler_counter_instance_id_t id, rocprofiler_counter_dimension_id_t dim, unsigned long *pos);

rocprofiler_flush_buffer_t rocprofiler_flush_buffer_FPTR;
rocprofiler_sample_device_counting_service_t rocprofiler_sample_device_counting_service_FPTR;
rocprofiler_configure_callback_dispatch_counting_service_t rocprofiler_configure_callback_dispatch_counting_service_FPTR;
rocprofiler_configure_device_counting_service_t rocprofiler_configure_device_counting_service_FPTR;
rocprofiler_create_buffer_t rocprofiler_create_buffer_FPTR;
rocprofiler_create_context_t rocprofiler_create_context_FPTR;
rocprofiler_start_context_t rocprofiler_start_context_FPTR;
rocprofiler_stop_context_t rocprofiler_stop_context_FPTR;
rocprofiler_context_is_active_t rocprofiler_context_is_active_FPTR;
rocprofiler_context_is_valid_t rocprofiler_context_is_valid_FPTR;
rocprofiler_create_profile_config_t rocprofiler_create_profile_config_FPTR;
rocprofiler_force_configure_t rocprofiler_force_configure_FPTR;
rocprofiler_get_status_string_t rocprofiler_get_status_string_FPTR;
rocprofiler_get_thread_id_t rocprofiler_get_thread_id_FPTR;
rocprofiler_is_finalized_t rocprofiler_is_finalized_FPTR;
rocprofiler_is_initialized_t rocprofiler_is_initialized_FPTR;
rocprofiler_iterate_agent_supported_counters_t rocprofiler_iterate_agent_supported_counters_FPTR;
rocprofiler_iterate_counter_dimensions_t rocprofiler_iterate_counter_dimensions_FPTR;
rocprofiler_query_available_agents_t rocprofiler_query_available_agents_FPTR;
rocprofiler_query_counter_info_t rocprofiler_query_counter_info_FPTR;
rocprofiler_query_counter_instance_count_t rocprofiler_query_counter_instance_count_FPTR;
rocprofiler_query_record_counter_id_t rocprofiler_query_record_counter_id_FPTR;
rocprofiler_query_record_dimension_position_t rocprofiler_query_record_dimension_position_FPTR;


/* ** */
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t client_ctx;
    return client_ctx;
}

rocprofiler_buffer_id_t&
get_buffer()
{
    static rocprofiler_buffer_id_t buf = {};
    return buf;
}

std::string
get_error_string()
{
    return _rocp_sdk_error_string;
}

void
set_error_string(std::string str)
{
    _rocp_sdk_error_string = str;
}

int
get_profiling_mode(void)
{
    return rpsdk_profiling_mode;
}

/* ** */
static const char *
obtain_function_pointers()
{
    static bool first_time = true;
    void *dllHandle = nullptr;

    if( !first_time )
        return NULL;

    const char* pathname = std::getenv("PAPI_ROCP_SDK_LIB");

    // If the user gave us an explicit path to librocprofiler-sdk.so, use it.
    if ( nullptr != pathname && strlen(pathname) <= PATH_MAX ) {
        dllHandle = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    }

    // If we were not given an explicit path, or the path didn't work, try elsewhere.
    if ( NULL == pathname || nullptr == dllHandle ) {
        std::string path2;
        const char *rocm_root = std::getenv("PAPI_ROCP_SDK_ROOT");
        if( nullptr == rocm_root || strlen(rocm_root) > PATH_MAX ){
            set_error_string("Did not find path for librocprofiler-sdk.so. Set either PAPI_ROCP_SDK_ROOT, or ROCP_SDK_LIB.");
            return get_error_string().c_str();
        }
        path2 = std::string(rocm_root) + "/lib/librocprofiler-sdk.so";

        // Clear previous errors.
        (void)dlerror();

        dllHandle = dlopen(path2.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (dllHandle == NULL) {
            set_error_string(std::string("Could not dlopen() librocprofiler-sdk.so. Set either PAPI_ROCP_SDK_ROOT, or ROCP_SDK_LIB. Error: ")+dlerror());
            return dlerror();
        }
    }

    DLL_SYM_CHECK(rocprofiler_flush_buffer, rocprofiler_flush_buffer_t);
    DLL_SYM_CHECK(rocprofiler_sample_device_counting_service, rocprofiler_sample_device_counting_service_t);
    DLL_SYM_CHECK(rocprofiler_configure_callback_dispatch_counting_service, rocprofiler_configure_callback_dispatch_counting_service_t);
    DLL_SYM_CHECK(rocprofiler_configure_device_counting_service, rocprofiler_configure_device_counting_service_t);
    DLL_SYM_CHECK(rocprofiler_create_context, rocprofiler_create_context_t);
    DLL_SYM_CHECK(rocprofiler_create_buffer, rocprofiler_create_buffer_t);
    DLL_SYM_CHECK(rocprofiler_start_context, rocprofiler_start_context_t);
    DLL_SYM_CHECK(rocprofiler_stop_context, rocprofiler_stop_context_t);
    DLL_SYM_CHECK(rocprofiler_context_is_valid, rocprofiler_context_is_valid_t);
    DLL_SYM_CHECK(rocprofiler_context_is_active, rocprofiler_context_is_active_t);
    DLL_SYM_CHECK(rocprofiler_create_profile_config, rocprofiler_create_profile_config_t);
    DLL_SYM_CHECK(rocprofiler_force_configure, rocprofiler_force_configure_t);
    DLL_SYM_CHECK(rocprofiler_get_status_string, rocprofiler_get_status_string_t);
    DLL_SYM_CHECK(rocprofiler_get_thread_id, rocprofiler_get_thread_id_t);
    DLL_SYM_CHECK(rocprofiler_is_finalized, rocprofiler_is_finalized_t);
    DLL_SYM_CHECK(rocprofiler_is_initialized, rocprofiler_is_initialized_t);
    DLL_SYM_CHECK(rocprofiler_iterate_agent_supported_counters, rocprofiler_iterate_agent_supported_counters_t);
    DLL_SYM_CHECK(rocprofiler_iterate_counter_dimensions, rocprofiler_iterate_counter_dimensions_t);
    DLL_SYM_CHECK(rocprofiler_query_available_agents, rocprofiler_query_available_agents_t);
    DLL_SYM_CHECK(rocprofiler_query_counter_info, rocprofiler_query_counter_info_t);
    DLL_SYM_CHECK(rocprofiler_query_counter_instance_count, rocprofiler_query_counter_instance_count_t);
    DLL_SYM_CHECK(rocprofiler_query_record_counter_id, rocprofiler_query_record_counter_id_t);
    DLL_SYM_CHECK(rocprofiler_query_record_dimension_position, rocprofiler_query_record_dimension_position_t);

    // Make sure we don't run this code multiple times.
    first_time = false;

    return NULL;
}

/**
 * For a given counter, query the dimensions that it has.
 */
std::vector<rocprofiler_record_dimension_info_t>
counter_dimensions(rocprofiler_counter_id_t counter)
{
    std::vector<rocprofiler_record_dimension_info_t> dims;
    rocprofiler_available_dimensions_cb_t cb;

    cb = [](rocprofiler_counter_id_t, const rocprofiler_record_dimension_info_t* dim_info, size_t num_dims, void *user_data) {
             std::vector<rocprofiler_record_dimension_info_t>* vec;

             vec = static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
             for(size_t i = 0; i < num_dims; i++){
                 vec->push_back(dim_info[i]);
             }
             return ROCPROFILER_STATUS_SUCCESS;
         };

    // Use the callback defined above to populate the vector "dims" with the dimension info of counter "counter".
    ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions_FPTR(counter, cb, &dims),
                     "Could not iterate counter dimensions");
    return dims;
}

/* ** */
bool
dimensions_match( dim_vector_t dim_instances, dim_vector_t recorded_dims )
{
    // Traverse all the dimensions in the event instance (i.e. base_event+qualifiers) of an event in the active_event_set_ctx
    for(const auto &ev_inst_dim : dim_instances ){
        bool found_dim_id = false;
        // Traverse all the dimensions of the event in the record_callback() data
        for(const auto &recorded_dim : recorded_dims ){
            if( ev_inst_dim.first == recorded_dim.first ){
                found_dim_id = true;
                // If the ids of two dimensions match, we compare the positions.
                if( ev_inst_dim.second != recorded_dim.second ){
                    return false;
                }
                // If we found a match, we don't need to check the remaining recorded dimensions against this qualifier.
                break;
            }
        }
        // if the record_callback() data does not have one of the dimensions of the event instance, then they didn't match.
        if( !found_dim_id ){
            return false;
        }
    }
    return true;
}

/* ** */
void
record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_record_counter_t*                record_data,
                size_t                                       record_count,
                rocprofiler_user_data_t,
                void*                                        callback_data_args)
{
    rec_info_t *tmp_rec_info;
    uint64_t device;

    if( (NULL == _counter_values) || (NULL == active_event_set_ctx) || (0 == (active_event_set_ctx->state & RPSDK_AES_RUNNING)) ){
        return;
    }

    _papi_hwi_lock(_rocp_sdk_lock);

    // Find the logical GPU id of this dispatch.
    auto agent = gpu_agents.find( dispatch_data.dispatch_info.agent_id.handle );
    if( gpu_agents.end() != agent ){
        device = agent->second->logical_node_type_id;
    }else{
        device = -1;
        SUBDBG("agent_id in dispatch_data does not correspond to a known gpu agent.\n");
    }

    // Create the mapping from events in the eventset (passed by the user) to entries in the "record_data" array.
    // The order of the entries in "record_data" will remain the same for a given profile, so we only need to do this once.
    if( index_mapping.empty() ){
        rec_info_t event_set_to_rec_mapping[record_count];

        index_mapping.resize( record_count*(active_event_set_ctx->num_events), false );

        // Traverse all the recorded entries and cache some information about them
        // that we will need further down when doing the matching.
        for(int i=0; i<record_count; ++i){
            rocprofiler_counter_id_t counter_id;
            rec_info_t &rec_info = event_set_to_rec_mapping[i];

            rec_info.device = device;

            ROCPROFILER_CALL(rocprofiler_query_record_counter_id_FPTR(record_data[i].id, &counter_id), "Could not retrieve counter_id");
            rec_info.counter_id = counter_id;

            std::vector<rocprofiler_record_dimension_info_t> dimensions = counter_dimensions(counter_id);
            for(auto& dim : dimensions ){
                unsigned long pos=0;
                ROCPROFILER_CALL(rocprofiler_query_record_dimension_position_FPTR(record_data[i].id, dim.id, &pos), "Count not retrieve dimension");
                rec_info.recorded_dims.emplace_back( std::make_pair(dim.id, pos) );
            }
        }

        // Traverse all events in the active event set and find which recorded entry matches each one of them.
        for( int ei=0; ei<active_event_set_ctx->num_events; ei++ ){
            double counter_value_sum = 0.0;
            auto e_tmp = papi_id_to_event_instance.find( active_event_set_ctx->event_ids[ei] );
            if( papi_id_to_event_instance.end() == e_tmp ){
                continue;
            }
            event_instance_info_t e_inst = e_tmp->second;

            for(int i=0; i<record_count; ++i){
                rec_info_t &rec_info = event_set_to_rec_mapping[i];
                if( ( e_inst.device != rec_info.device ) ||
                    ( e_inst.counter_info.id.handle != rec_info.counter_id.handle ) ||
                    !dimensions_match(e_inst.dim_instances, rec_info.recorded_dims)
                  ){
                    continue;
                }
                index_mapping[ei*record_count+i] = true;
            }
        }
    }

    // Traverse all events in the active event set and find which recorded entry matches each one of them.
    for( int ei=0; ei<active_event_set_ctx->num_events; ei++ ){
        double counter_value_sum = 0.0;

        for(int i=0; i<record_count; ++i){
            // All counters in the sample whose dimemsions match the qualifers of the event instance
            // will be added. This means that if a qualifier is missing, we will get the sum.
            if( true == index_mapping[ei*record_count+i] ){
                counter_value_sum += record_data[i].counter_value;
            }
        }
        // Rocprofiler-SDK default behavior in dispatch mode is to only report
        // the value of the counters since the dispatch of the kernel. However,
        // PAPI semantics dictate that counter values are only reset by
        // PAPI_reset(), etc, not by kernel invocations. Therefore, we must
        // accumulate the counter values (+=), not overwrite the previous one.
	_counter_values[ei] += counter_value_sum;
    }

    _papi_hwi_unlock(_rocp_sdk_lock);

#if defined(DEBUG_OUTPUT_OF_RECORDED_VALUES)
    for(size_t i = 0; i < record_count; ++i){
        rocprofiler_counter_id_t counter_id;
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id_FPTR(record_data[i].id, &counter_id), "Could not retrieve counter_id");
        std::cerr << " ## record_data[" << i << "].id: " << record_data[i].id << " -> counter_id: " << counter_id.handle << " Value= " << record_data[i].counter_value << std::endl;
    }
#endif
    return;
}


/* ** */
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_profile_config_id_t             *config,
                  rocprofiler_user_data_t *,
                  void * )
{

    // All threads get a shared lock because if they are only reading the
    // existing config from the cache they can all do this at the same
    // time. If there is nothing in the cache, they will exit this scope
    // and the lock will be automatically released.
    const SHARED_LOCK rlock(profile_cache_mutex);

    auto pos = rpsdk_profile_cache.find(dispatch_data.dispatch_info.agent_id.handle);
    if( rpsdk_profile_cache.end() != pos ){
        *config = pos->second;
    }
    return;

}

/* ** */
agent_map_t
get_GPU_agent_info() {
    auto iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                         const void**                agents_arr,
                         size_t                      num_agents,
                         void*                       user_data) {

        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};

        auto* agents_v = static_cast<agent_map_t*>(user_data);
        for(size_t i = 0; i < num_agents; ++i) {
            const auto* itr = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if( ROCPROFILER_AGENT_TYPE_GPU == itr->type ){
                agents_v->emplace(itr->id.handle, itr);
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto _agents = agent_map_t{};
    ROCPROFILER_CALL( rocprofiler_query_available_agents_FPTR(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           static_cast<void*>(&_agents)),
                      "query available agents");

    return _agents;
}

/* ** */
void
set_profile(rocprofiler_context_id_t                 context_id,
            rocprofiler_agent_id_t                   agent,
            rocprofiler_agent_set_profile_callback_t set_config,
            void*)
{
    const SHARED_LOCK rlock(profile_cache_mutex);

    auto pos = rpsdk_profile_cache.find(agent.handle);
    if( rpsdk_profile_cache.end() != pos ){
        set_config(context_id, pos->second);
    }
    return;
}

/* ** */
void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*                         user_data,
                  uint64_t)
{
	return;
}

/* ** */
int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{


    assert(tool_data != nullptr);

    ROCPROFILER_CALL(rocprofiler_create_context_FPTR(&get_client_ctx()), "context creation");

    if( RPSDK_MODE_DEVICE_SAMPLING == get_profiling_mode() ){
        ROCPROFILER_CALL(rocprofiler_create_buffer_FPTR(get_client_ctx(),
                                               32*1024,
                                               16*1024,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               buffered_callback,
                                               tool_data,
                                               &get_buffer()),
                         "buffer creation failed");
    }else{
        ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service_FPTR(
                             get_client_ctx(), dispatch_callback, tool_data, record_callback, tool_data),
                         "Could not setup callback dispatch");
    }

    return 0;

}

/* ** */
static void
delete_event_list(void){
    base_events_by_name.clear();
}

/* ** */
static int
assign_id_to_event(std::string event_name, event_instance_info_t ev_inst_info){
    int papi_event_id = -1;

    // Note: _global_papi_event_count is std::atomic, so the followign line is thread safe.
    papi_event_id = _global_papi_event_count++;
    papi_id_to_event_instance[ papi_event_id ] = ev_inst_info;
    event_instance_name_to_papi_id[ event_name ] = papi_event_id;

    return papi_event_id;
}

/* ** */
static void
populate_event_list(void){

    // If the event list is already populated, return without doing anything.
    if( !base_events_by_name.empty() )
        return;

    // Pick the first agent, because we currently do not support a mixture of heterogeneous GPUs, so all agents should be the same.
    const rocprofiler_agent_v0_t *agent = gpu_agents.begin()->second;

    // GPU Counter IDs
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    auto itrt_cntr_cb = [](rocprofiler_agent_id_t,
                           rocprofiler_counter_id_t* counters,
                           size_t                    num_counters,
                           void*                     udata) {
                               std::vector<rocprofiler_counter_id_t>* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(udata);
                               for(size_t i = 0; i < num_counters; i++){
                                   vec->push_back(counters[i]);
                               }
                               return ROCPROFILER_STATUS_SUCCESS;
                           };

    // Get the counters available through the selected agent.
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters_FPTR(agent->id, itrt_cntr_cb, static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    for(auto& counter : gpu_counters){
        rocprofiler_counter_info_v0_t counter_info;
        ROCPROFILER_CALL(
             rocprofiler_query_counter_info_FPTR(counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&counter_info)),
            "Could not query info");

        std::vector<rocprofiler_record_dimension_info_t> dim_info;
        dim_info = counter_dimensions(counter_info.id);

        base_events_by_name[counter_info.name].counter_info = counter_info;
        base_events_by_name[counter_info.name].dim_info = dim_info;

        ++_base_event_count;
        // This list does not contain "proper" events, with all qualifiers that
        // PAPI requires. This is just the list of base events as enumerated by the
        // vendor API. Therefore, it's ok to set "dim_instances" and "device" to dummy values.
        event_instance_info_t ev_inst_info;
        ev_inst_info.qualifiers_present = 0;
        ev_inst_info.event_inst_name = counter_info.name;
        ev_inst_info.counter_info = counter_info;
        ev_inst_info.dim_info = dim_info;
        ev_inst_info.dim_instances = {};
        ev_inst_info.device = -1;
        (void)assign_id_to_event(counter_info.name, ev_inst_info);
    }

    return;

}

/* ** */
void stop_counting(void){
    int ctx_active, ctx_valid;
    _counter_values = NULL;
    ROCPROFILER_CALL(rocprofiler_context_is_valid_FPTR(get_client_ctx(), &ctx_valid), "check context validity");
    if( !ctx_valid ){
        SUBDBG("client_context is invalid\n");
        return;
    }
    ROCPROFILER_CALL(rocprofiler_context_is_active_FPTR(get_client_ctx(), &ctx_active), "check if context is active");
    if( !ctx_active ){
        SUBDBG("client_context is not active\n");
        return;
    }
    ROCPROFILER_CALL(rocprofiler_stop_context_FPTR(get_client_ctx()), "stop context");
}

/* ** */
void
start_counting(vendorp_ctx_t ctx){
    static bool is_device_counting_configured = false;

    // Store a pointer to the counter value array in a global variable so that
    // our functions that are called from the ROCprofiler-SDK (instead of our
    // API) can still find the array.
    _counter_values = ctx->counters;

    if( (RPSDK_MODE_DEVICE_SAMPLING == get_profiling_mode()) && !is_device_counting_configured ){
        is_device_counting_configured = true;
        // Configure device_counting_service for all devices.
        for(auto g_it=gpu_agents.begin(); g_it!=gpu_agents.end(); ++g_it){
            ROCPROFILER_CALL(rocprofiler_configure_device_counting_service_FPTR(
                                 get_client_ctx(), get_buffer(), g_it->second->id, set_profile, nullptr),
                             "Could not setup sampling");
        }
    }

    ROCPROFILER_CALL(rocprofiler_start_context_FPTR(get_client_ctx()), "start context");
}

/* ** */
int
read_sample(){
    int papi_errno = PAPI_OK;
    int ret_val;
    size_t rec_count = 1024;
    rocprofiler_record_counter_t output_records[1024];

    if( (NULL == _counter_values) || (NULL == active_event_set_ctx) || (0 == (active_event_set_ctx->state & RPSDK_AES_RUNNING)) ){
        papi_errno = PAPI_ENOTRUN;
        goto fn_fail;
    }

    ret_val = rocprofiler_sample_device_counting_service_FPTR(
                get_client_ctx(), {}, ROCPROFILER_COUNTER_FLAG_NONE,
                output_records, &rec_count);

    if( ret_val != ROCPROFILER_STATUS_SUCCESS ){
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    // Create the mapping from events in the eventset (passed by the user) to entries (samples) in the "output_records" array.
    // The order of the entries in "output_records" will remain the same for a given profile, so we only need to do this once.
    if( index_mapping.empty() ){
        rec_info_t event_set_to_rec_mapping[rec_count];

        index_mapping.resize( rec_count*(active_event_set_ctx->num_events), false );

        // Traverse all the sampled entries and cache some information about them
        // that we will need further down when doing the matching.
        for(int i=0; i<rec_count; ++i){
            rocprofiler_counter_id_t counter_id;
            rec_info_t &rec_info = event_set_to_rec_mapping[i];

            auto agent = gpu_agents.find( output_records[i].agent_id.handle );
            if( gpu_agents.end() != agent ){
                rec_info.device = agent->second->logical_node_type_id;
            }else{
                SUBDBG("agent_id of recorded sample %d does not correspond to a known gpu agent.\n", i);
            }

            ROCPROFILER_CALL(rocprofiler_query_record_counter_id_FPTR(output_records[i].id, &counter_id), "Could not retrieve counter_id");
            rec_info.counter_id = counter_id;

#if defined(DEBUG_OUTPUT)
            printf(" ## output_records[%d].id: %lu -> counter_id: %lu Value= %lf\n", i, output_records[i].id, counter_id.handle, output_records[i].counter_value);
	    fflush(stdout);
#endif // DEBUG_OUTPUT

            std::vector<rocprofiler_record_dimension_info_t> dimensions = counter_dimensions(counter_id);
            for(auto& dim : dimensions ){
                unsigned long pos=0;
                ROCPROFILER_CALL(rocprofiler_query_record_dimension_position_FPTR(output_records[i].id, dim.id, &pos), "Count not retrieve dimension");
                rec_info.recorded_dims.emplace_back( std::make_pair(dim.id, pos) );
            }
        }

        // Traverse all events in the active event set and find which entries in the set of samples matches each one of them.
        for( int ei=0; ei<active_event_set_ctx->num_events; ei++ ){
            double counter_value_sum = 0.0;

            // Find the internal event instance.
            auto tmp = papi_id_to_event_instance.find( active_event_set_ctx->event_ids[ei] );
            if( papi_id_to_event_instance.end() == tmp ){
                SUBDBG("EventSet contains an event id that is unknown to the rocp_sdk component.\n");
                continue;
            }
            event_instance_info_t e_inst = tmp->second;

            for(int i=0; i<rec_count; ++i){
                rec_info_t &rec_info = event_set_to_rec_mapping[i];
                if( ( e_inst.device != rec_info.device ) ||
                    ( e_inst.counter_info.id.handle != rec_info.counter_id.handle ) ||
                    !dimensions_match(e_inst.dim_instances, rec_info.recorded_dims)
                  ){
                    continue;
                }

                index_mapping[ei*rec_count+i] = true;
            }
        }
    }

    // Traverse all events in the active event set and find which entry in the sample matches each one of them.
    for( int ei=0; ei<active_event_set_ctx->num_events; ei++ ){
        double counter_value_sum = 0.0;

        for(int i=0; i<rec_count; ++i){
            // All counters in the sample whose dimemsions match the qualifers of the event instance
            // will be added. This means that if a qualifier is missing, we will get the sum.
            if( true == index_mapping[ei*rec_count+i] ){
                counter_value_sum += output_records[i].counter_value;
            }
        }
        _counter_values[ei] = counter_value_sum;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

/* ** */
int
evt_id_to_descr(int event_id, const char **desc){

    auto it = papi_id_to_event_instance.find( event_id );
    if( papi_id_to_event_instance.end() == it ){
        return PAPI_ENOEVNT;
    }
    *desc = it->second.counter_info.description;

    return PAPI_OK;
}

/* ** */
int
evt_id_to_name(int papi_event_id, const char **name){

    auto it = papi_id_to_event_instance.find( papi_event_id );
    if( papi_id_to_event_instance.end() == it ){
        return PAPI_ENOEVNT;
    }
    *name = it->second.event_inst_name.c_str();

    return PAPI_OK;
}


/* ** */
static int
build_event_info_from_name(std::string event_name, event_instance_info_t *ev_inst_info){
    int pos=0, ppos=0;
    std::vector<std::string> qualifiers = {};
    dim_vector_t dim_instances = {};
    std::string base_event_name;
    uint64_t qualifiers_present = 0;
    int device_qualifier_value = -1;

    pos=event_name.find(':');
    if( pos == event_name.npos){
        base_event_name = event_name;
    }else{
        base_event_name = event_name.substr(0, pos-0);
        ppos = pos+1;
        // Tokenize the event name and keep the qualifiers in a vector.
        while( (pos=event_name.find(':', ppos)) != event_name.npos){
            std::string qual_tuple = event_name.substr(ppos,pos-ppos);
            qualifiers.emplace_back( qual_tuple );
            ppos = pos+1;
        }
        // Add in the vector the last qualifier we found in the while loop.
        qualifiers.emplace_back( event_name.substr(ppos,pos-ppos) );
    }

    auto it0 = base_events_by_name.find(base_event_name);
    if( base_events_by_name.end() == it0 ){
        return PAPI_ENOEVNT;
    }
    base_event_info_t base_event_info = it0->second;

    for( const auto & qual : qualifiers ){
        // All qualifiers must have the form "qual_name=qual_value".
        pos=qual.find('=');
        if( pos == qual.npos){
            return PAPI_EINVAL;
        }

        std::string qual_name = qual.substr(0, pos-0);
        int qual_val = std::stoi( qual.substr(pos+1) );

        // The "device" qualifier does not appear as a rocprofiler-sdk dimension.
        // It comes from us (the PAPI component), so it needs special treatment.
        if( qual_name.compare("device") == 0 ){
            // We use the most significant bit to designate the presence of the "device" qualifier.
            qualifiers_present |= (1 << base_event_info.dim_info.size());
            device_qualifier_value = qual_val;
        }else{
            int qual_i = 0;
            // Make sure that the qualifier name corresponds to one of the known dimensions of this event.
            for( const auto & dim : base_event_info.dim_info ){
                if( qual_name.compare(dim.name) == 0 ){
                    // Make sure that the qualifier value is within the proper range.
                    if( qual_val >= dim.instance_size ){
                        return PAPI_EINVAL;
                    }
                    dim_instances.emplace_back( std::make_pair(dim.id, qual_val) );
                    // Mark which qualifiers we have found based on the order in which they appear in
                    // base_event_info.dim_info, NOT based on the order the user provided them.
                    // This will work up to 64 possible qualifiers.
                    if( qual_i < 64 ){
                        qualifiers_present |= (1 << qual_i);
                    }else{
                        SUBDBG("More than 64 qualifiers detected in event name: %s\n",event_name.c_str());
                    }
                }
                ++qual_i;
            }
        }

    }

    // Sort the qualifiers (dimension instances) based on dimension id. This allows the user to give us the
    // qualifiers in any order.
    std::sort(dim_instances.begin(), dim_instances.end(),
              [](const dim_t &a, const dim_t &b) { return (a.first < b.first); }
             );

    ev_inst_info->qualifiers_present = qualifiers_present;
    ev_inst_info->event_inst_name = event_name;
    ev_inst_info->counter_info = base_event_info.counter_info;
    ev_inst_info->dim_info = base_event_info.dim_info;
    ev_inst_info->dim_instances = dim_instances;
    ev_inst_info->device = device_qualifier_value;

    return PAPI_OK;
}

int
evt_name_to_id(std::string event_name, unsigned int *event_id){
    int ret_val = PAPI_OK;
    event_instance_info_t ev_inst_info;
    unsigned int papi_event_id;

    // If the event already exists in our metadata, return its id.
    auto it1 = event_instance_name_to_papi_id.find( event_name );
    if( event_instance_name_to_papi_id.end() != it1 ){
        papi_event_id = it1->second;
    }else{
        // If we've never seen this event before, insert the info into our metadata.
        ret_val = build_event_info_from_name(event_name, &ev_inst_info);
        if( PAPI_OK != ret_val ){
            return ret_val;
        }
        papi_event_id = assign_id_to_event(event_name, ev_inst_info);
    }

    *event_id = papi_event_id;
    return PAPI_OK;
}

/* ** */
int
evt_enum(unsigned int *event_code, int modifier){
    int papi_errno=PAPI_OK, tmp_code;
    base_event_info_t event_info;
    std::string full_name;
    event_instance_info_t ev_inst;

    populate_event_list();

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            papi_errno = PAPI_OK;
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            tmp_code = *event_code + 1;
            if( tmp_code >= _base_event_count ){
                papi_errno = PAPI_ENOEVNT;
                break;
            }
            papi_errno = PAPI_OK;
            *event_code = tmp_code;

            break;
        case PAPI_NTV_ENUM_UMASKS:
            tmp_code = *event_code;

            {
                std::string qual_ub, tmp_desc;
                auto it = papi_id_to_event_instance.find( tmp_code );
                if( papi_id_to_event_instance.end() == it ){
                    papi_errno = PAPI_ENOEVNT;
                    break;
                }
                ev_inst = it->second;
                int qual_i=-1;
                // Find the last qualifier present so that we can create an event instance using the next qualifier in the list.
                for(int i=0; i<64; ++i){
                    if( ( ev_inst.qualifiers_present >> i) & 0x1 ){
                        qual_i = i;
                    }
                }
                // Increment the last one found by one to create the next potential qualifier index.
                ++qual_i;

                // If we exceeded the number of available dimensions (i.e. qualifiers) then we are done with this base event.
                if( qual_i > ev_inst.dim_info.size() ){
                    papi_errno = PAPI_ENOEVNT;
                    break;
                // Here we insert the "device" qualifier, which does not appear as a dimension in rocprofiler-sdk.
                }else if( qual_i == ev_inst.dim_info.size() ){
                    full_name = ev_inst.counter_info.name + std::string(":device=0");
                    qual_ub = std::to_string(gpu_agents.size()-1);
                    tmp_desc = "masks: Range: [0-" + qual_ub + "], default=0.";
                }else{
                    full_name = ev_inst.counter_info.name + std::string(":device=0");
                    rocprofiler_record_dimension_info_t dim = ev_inst.dim_info[qual_i];
                    full_name = ev_inst.counter_info.name + std::string(":") + dim.name + std::string("=0");
                    qual_ub = std::to_string(dim.instance_size-1);
                    tmp_desc = "masks: Range: [0-" + qual_ub + "], default=sum.";
                }

                // Insert the new event (base_event:SOME_QUALIFIER=0) into the data structures and get an event_code for it.
                evt_name_to_id(full_name, event_code);
                papi_id_to_event_instance[*event_code].counter_info.description = strdup(tmp_desc.c_str());

                papi_errno = PAPI_OK;
                break;

            }

        default:
            papi_errno = PAPI_EINVAL;
            break;
    }

    return papi_errno;
}

/* ** */
void
empty_active_event_set(void){
    active_event_set_ctx = NULL;

    index_mapping.clear();

    active_device_set.clear();
    return;
}

/* ** */
int
set_profile_cache(vendorp_ctx_t ctx){
    std::map<uint64_t, std::vector<event_instance_info_t> > active_events_per_device;

    // Acquire a unique lock so that no other thread can try to read
    // the profile cache while we are modifying it.
    const UNIQUE_LOCK wlock(profile_cache_mutex);

    rpsdk_profile_cache.clear();

    for( int i=0; i < ctx->num_events; ++i) {
        // make sure the event exists.
        auto it = papi_id_to_event_instance.find( ctx->event_ids[i] );
        if( papi_id_to_event_instance.end() == it ){
            return PAPI_ENOEVNT;
        }

        active_device_set.insert(it->second.device);
        active_events_per_device[it->second.device].emplace_back(it->second);
    }

    for(const auto &a_it : gpu_agents ){
        rocprofiler_profile_config_id_t profile;

        auto agent = a_it.second;

        std::vector<rocprofiler_counter_id_t> event_vid_list = {};
        std::set<uint64_t> id_set = {};

        for( const auto e_inst : active_events_per_device[agent->logical_node_type_id] ){

            rocprofiler_counter_id_t vid = e_inst.counter_info.id;
            // If the vid of the event (base event) is not already in the event_vid_list, then add it.
            if( id_set.find(vid.handle) == id_set.end() ){
                event_vid_list.emplace_back( vid );
                id_set.emplace( vid.handle );
            }
        }

        //TODO Error handling: right now we can't tell which event caused the problem, if a problem occurs.
        ROCPROFILER_CALL(rocprofiler_create_profile_config_FPTR(agent->id,
                                                           event_vid_list.data(),
                                                           event_vid_list.size(),
                                                           &profile),
                         "Could not construct profile cfg");

        rpsdk_profile_cache.emplace(agent->id.handle, profile);
    }

    return PAPI_OK;
}

/* ** */
void tool_fini(void* tool_data) {
    stop_counting();
    empty_active_event_set();
    return;
}

/* ** */
int setup() {
    int status = 0;

    // Set sampling as the default mode and allow the users to change this
    // behavior by setting the environment variable PAPI_ROCP_SDK_DISPATCH_MODE
    rpsdk_profiling_mode = RPSDK_MODE_DEVICE_SAMPLING;
    if( NULL != getenv("PAPI_ROCP_SDK_DISPATCH_MODE") ){
        rpsdk_profiling_mode = RPSDK_MODE_DISPATCH;
    }

    const char *error_msg = obtain_function_pointers();
    if( NULL != error_msg ){
        set_error_string("Could not obtain all functions from librocprofiler-sdk.so. Possible library version mismatch.");
        SUBDBG("dlsym(): %s\n", error_msg);
        goto fn_fail;
    }

    // Obtain the list of available (GPU) agents.
    gpu_agents = get_GPU_agent_info();

    if( (ROCPROFILER_STATUS_SUCCESS == rocprofiler_is_initialized_FPTR(&status)) && (0 == status) ){
        ROCPROFILER_CALL(rocprofiler_force_configure_FPTR(&rocprofiler_configure), "force configuration");
    }

    return PAPI_OK;

  fn_fail:
    return PAPI_ECMP;
}

}  // namespace papi_rocpsdk

//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------

extern "C" int
rocprofiler_sdk_init_pre(void)
{
    return PAPI_OK;
}

extern "C" int
rocprofiler_sdk_init(void)
{
    int papi_errno=PAPI_OK;

    if( papi_rocpsdk::setup() ){
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    papi_rocpsdk::populate_event_list();

  fn_exit:
    return papi_errno;
  fn_fail:
      papi_rocpsdk::delete_event_list();
    goto fn_exit;
}

extern "C" int
rocprofiler_sdk_shutdown(void)
{
    papi_rocpsdk::stop_counting();
    papi_rocpsdk::empty_active_event_set();
    papi_rocpsdk::delete_event_list();
    return PAPI_OK;
}

extern "C" int
rocprofiler_sdk_stop(vendorp_ctx_t ctx)
{
    if( ctx ){
        ctx->state = RPSDK_AES_STOPPED;
    }

    finalize_ctx(ctx);
    papi_rocpsdk::stop_counting();
    papi_rocpsdk::empty_active_event_set();
    papi_rocpsdk::delete_event_list();
    return PAPI_OK;
}

extern "C" int
rocprofiler_sdk_start(vendorp_ctx_t ctx)
{
    int i;
    for(i=0; i<ctx->num_events; i++){
        ctx->counters[i] = 0;
    }
    papi_rocpsdk::start_counting(ctx);

    ctx->state |= RPSDK_AES_RUNNING;

    return PAPI_OK;
}

extern "C" int
rocprofiler_sdk_ctx_reset(vendorp_ctx_t ctx)
{
    int i;
    if( !ctx ){
        SUBDBG("Trying to reset a component before calling PAPI_start().");
        return PAPI_EINVAL;
    }

    for(i=0; i<ctx->num_events; i++){
        ctx->counters[i] = 0;
    }

    return PAPI_OK;
}

extern "C" int
rocprofiler_sdk_ctx_open(int *event_ids, int num_events, vendorp_ctx_t *ctx)
{
    int papi_errno=PAPI_OK;

    *ctx = (vendorp_ctx_t)papi_calloc(1, sizeof(struct vendord_ctx));
    if (NULL == *ctx) {
        return PAPI_ENOMEM;
    }

    _papi_hwi_lock(_rocp_sdk_lock);

    papi_rocpsdk::empty_active_event_set();
    papi_errno = init_ctx(event_ids, num_events, *ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    papi_rocpsdk::active_event_set_ctx = *ctx;
    papi_rocpsdk::set_profile_cache(*ctx);

    (*ctx)->state = RPSDK_AES_OPEN;

  fn_exit:
    _papi_hwi_unlock(_rocp_sdk_lock);
    return papi_errno;
  fn_fail:
    finalize_ctx(*ctx);
    goto fn_exit;
}

extern "C" int
rocprofiler_sdk_ctx_read(vendorp_ctx_t ctx, long long **counters)
{
    int papi_errno = PAPI_OK;

    // If the collection mode is DEVICE_SAMPLING get an explicit sample.
    if( RPSDK_MODE_DEVICE_SAMPLING == papi_rocpsdk::get_profiling_mode() ){
        papi_errno = papi_rocpsdk::read_sample();
    }

    // If the mode is not sampling the counter data should already be in
    // "ctx->counters", because record_callback() should have placed it there
    // asynchronously.
    // However, we don't use any synchronization mechanisms to guarantee that
    // record_callback() has indeed completed before this function returns.
    // Therefore, we recomend that the user code adds a small delay between
    // the completion of a GPU kernel and the call of PAPI_read().

    *counters = ctx->counters;
    return papi_errno;
}

extern "C" int
rocprofiler_sdk_evt_enum(unsigned int *event_code, int modifier)
{
    return papi_rocpsdk::evt_enum(event_code, modifier);
}

extern "C" int
rocprofiler_sdk_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = PAPI_OK;
    const char *tmp_name;

    papi_errno = papi_rocpsdk::evt_id_to_name(event_code, &tmp_name);
    if( PAPI_OK == papi_errno ){
        snprintf(name, len, "%s", tmp_name);
    }

    return papi_errno;
}

extern "C" int
rocprofiler_sdk_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno = PAPI_OK;
    const char *tmp_descr;

    papi_errno = papi_rocpsdk::evt_id_to_descr(event_code, &tmp_descr);
    if ( PAPI_OK == papi_errno ) {
        snprintf(descr, len, "%s", tmp_descr);
    }

    return papi_errno;
}

extern "C" int
rocprofiler_sdk_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    int papi_errno = PAPI_OK;
    const char *tmp_name, *tmp_descr;

    papi_errno = papi_rocpsdk::evt_id_to_name(event_code, &tmp_name);
    if ( PAPI_OK == papi_errno ) {
        snprintf(info->symbol, PAPI_HUGE_STR_LEN, "%s", tmp_name);
    }

    papi_errno = papi_rocpsdk::evt_id_to_descr(event_code, &tmp_descr);
    if ( PAPI_OK == papi_errno ) {
        snprintf(info->long_descr, PAPI_HUGE_STR_LEN, "%s", tmp_descr);
        snprintf(info->short_descr, PAPI_MIN_STR_LEN, "%s", tmp_descr);
    }

    return papi_errno;
}

extern "C" int
rocprofiler_sdk_evt_name_to_code(const char *event_name, unsigned int *event_code)
{
    int papi_errno = PAPI_OK;

    // If "device" qualifier is not provived by the user, make it zero.
    if( NULL == strstr(event_name, "device=") ){
        char *amended_event_name;
        size_t len = strlen(event_name)+strlen(":device=0")+1; // +1 for '\0'
        if( len > 1024 ){
            return PAPI_EMISC;
        }
        amended_event_name = (char *)calloc(len, sizeof(char));
        int ret = snprintf(amended_event_name, len, "%s:device=0", event_name);
        if( ret >= len ){
            free(amended_event_name);
            return PAPI_EMISC;
        }

        papi_errno = papi_rocpsdk::evt_name_to_id(amended_event_name, event_code);
    }else{
        papi_errno = papi_rocpsdk::evt_name_to_id(event_name, event_code);
    }

    return papi_errno;
}

extern "C" int
rocprofiler_sdk_err_get_last(const char **err){
    *err = strdup(papi_rocpsdk::get_error_string().substr(0,PAPI_MAX_STR_LEN-1).c_str() );
    return PAPI_OK;
}

static int
init_ctx(int *event_ids, int num_events, vendorp_ctx_t ctx)
{
    ctx->event_ids = event_ids;
    ctx->num_events = num_events;
    ctx->counters = (long long *)papi_calloc(num_events, sizeof(long long));
    if (NULL == ctx->counters) {
        return PAPI_ENOMEM;
    }
    return PAPI_OK;
}

static int
finalize_ctx(vendorp_ctx_t ctx)
{
    if( ctx ){
        ctx->event_ids = NULL;
        ctx->num_events = 0;
        free(ctx->counters);
        ctx->counters = NULL;
    }
    free(ctx);
    return PAPI_OK;
}

rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    const char *error_msg = papi_rocpsdk::obtain_function_pointers();

    if( NULL != error_msg ){
        papi_rocpsdk::set_error_string("Could not obtain all functions from librocprofiler-sdk.so. Possible library version mismatch.");
        SUBDBG("dlsym(): %s\n", error_msg);
        return NULL;
    }

    // set the client name
    id->name = "PAPI_ROCP_SDK_COMPONENT";

    auto* client_tool_data = new std::string("CLIENT_TOOL_STRING");

    // create configure data
    static auto cfg = rocprofiler_tool_configure_result_t{
                          sizeof(rocprofiler_tool_configure_result_t),
                          &papi_rocpsdk::tool_init,
                          &papi_rocpsdk::tool_fini,
                          static_cast<void*>(client_tool_data)
                      };

    // return pointer to configure data
    return &cfg;
}
