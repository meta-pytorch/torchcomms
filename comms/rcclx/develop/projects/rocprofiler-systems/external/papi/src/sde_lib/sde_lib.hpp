/**
 * @file    sde_lib.cpp
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  C++ API for SDEs.
 */

#include <type_traits>
#include <exception>

namespace papi_sde
{
    class PapiSde {
        private:
          papi_handle_t sde_handle;

        public:
          PapiSde(const char *name_of_library){
              sde_handle = papi_sde_init(name_of_library);
          }

          class CreatedCounter;
          class Recorder;
          class CountingSet;

          template <typename T>
          int register_counter(const char *event_name, int cntr_mode, T &counter ){
              if( std::is_same<long long int, T>::value )
                  return papi_sde_register_counter(sde_handle, event_name, cntr_mode, PAPI_SDE_long_long, &counter);
              if( std::is_same<int, T>::value )
                  return papi_sde_register_counter(sde_handle, event_name, cntr_mode, PAPI_SDE_int, &counter);
              if( std::is_same<double, T>::value )
                  return papi_sde_register_counter(sde_handle, event_name, cntr_mode, PAPI_SDE_double, &counter);
              if( std::is_same<float, T>::value )
                  return papi_sde_register_counter(sde_handle, event_name, cntr_mode, PAPI_SDE_float, &counter);
          }

          template <typename T, typename P>
          int register_counter_cb(const char *event_name, int cntr_mode, T (*func_ptr)(P*), P &param){
              if( std::is_same<long long int, T>::value ){
                  return papi_sde_register_fp_counter(sde_handle, event_name, cntr_mode, PAPI_SDE_long_long, (papi_sde_fptr_t)func_ptr, &param);
              }else{
                  SDE_ERROR("register_counter_cb() is currently limited to callback functions that have a return type of 'long long int'.");
                  return SDE_EINVAL;
              }
          }

          int unregister_counter(const char *event_name ){
              return papi_sde_unregister_counter(sde_handle, event_name);
          }

          int describe_counter(const char *event_name, const char *event_description ){
              return papi_sde_describe_counter(sde_handle, event_name, event_description);
          }

          int add_counter_to_group(const char *event_name, const char *group_name, uint32_t group_flags ){
              return papi_sde_add_counter_to_group(sde_handle, event_name, group_name, group_flags);
          }

          CreatedCounter *create_counter(const char *event_name, int cntr_mode){
               CreatedCounter *ptr;
               try{
                   ptr = new CreatedCounter(sde_handle, event_name, cntr_mode);
               }catch(std::exception const &e){
                   return nullptr;
               }
               return ptr;
          }

          Recorder *create_recorder(const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2)){
              Recorder *ptr;
              try{
                  ptr = new Recorder(sde_handle, event_name, typesize, cmpr_func_ptr);
               }catch(std::exception const &e){
                   return nullptr;
               }
               return ptr;
          }

          CountingSet *create_counting_set(const char *cset_name){
               CountingSet *ptr;
               try{
                   ptr = new CountingSet(sde_handle, cset_name);
               }catch(std::exception const &e){
                   return nullptr;
               }
               return ptr;
          }

        class Recorder {
            private:
              void *recorder_handle=nullptr;

            public:
              Recorder(papi_handle_t sde_handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2)){
                  if( SDE_OK != papi_sde_create_recorder(sde_handle, event_name, typesize, cmpr_func_ptr, &recorder_handle ) )
                      throw std::exception();
              }

              template <typename T>
              int record(T const &value){
                  if( nullptr != recorder_handle )
                      return papi_sde_record(recorder_handle, sizeof(T), &value);
                  else
                      return SDE_EINVAL;
              }

              int reset(void){
                  if( nullptr != recorder_handle )
                      return papi_sde_reset_recorder(recorder_handle);
                  else
                      return SDE_EINVAL;
              }
        };

        class CreatedCounter {
            private:
              void *counter_handle=nullptr;

            public:
              CreatedCounter(papi_handle_t sde_handle, const char *event_name, int cntr_mode){
                  if( SDE_OK != papi_sde_create_counter(sde_handle, event_name, cntr_mode, &counter_handle) )
                      throw std::exception();
              }

              template <typename T>
              int increment(T const &increment){
                  if( nullptr == counter_handle )
                      return SDE_EINVAL;

                  if( std::is_same<long long int, T>::value ){
                      return papi_sde_inc_counter(counter_handle, increment );
                  }else{
                      // for now we don't have the C API to handle increments other than "long long",
                      // but we can add this in the future transparently to the user.
                      return papi_sde_inc_counter(counter_handle, (long long int)increment );
                  }
              }

              int reset(void){
                  if( nullptr != counter_handle )
                      return papi_sde_reset_counter(counter_handle);
                  else
                      return SDE_EINVAL;
              }

        }; // class CreatedCounter

        class CountingSet {
            private:
              void *cset_handle=nullptr;

            public:
              CountingSet(papi_handle_t sde_handle, const char *cset_name){
                  if( SDE_OK != papi_sde_create_counting_set(sde_handle, cset_name, &cset_handle) )
                      throw std::exception();
              }

              template <typename T> int insert(T const &element, uint32_t type_id){
                  if( nullptr == cset_handle )
                      return SDE_EINVAL;
                  return papi_sde_counting_set_insert( cset_handle, sizeof(T), sizeof(T), &element, type_id);
              }

              template <typename T> int insert(size_t hashable_size, T const &element, uint32_t type_id){
                  if( nullptr == cset_handle )
                      return SDE_EINVAL;
                  return papi_sde_counting_set_insert( cset_handle, sizeof(T), hashable_size, &element, type_id);
              }

              template <typename T> int remove(size_t hashable_size, T const &element, uint32_t type_id){
                  if( nullptr == cset_handle )
                      return SDE_EINVAL;
                  return papi_sde_counting_set_remove( cset_handle, hashable_size, &element, type_id);
              }
        }; // class CountingSet

    }; // class PapiSde

    template <typename T>
    PapiSde::CreatedCounter &operator+=(PapiSde::CreatedCounter &X, const T increment){
        X.increment(increment);
        return X;
    }
    // Prefix increment ++x;
    inline PapiSde::CreatedCounter &operator++(PapiSde::CreatedCounter &X){
        X.increment(1LL);
        return X;
    }
    // Prefix decrement --x;
    inline PapiSde::CreatedCounter &operator--(PapiSde::CreatedCounter &X){
        X.increment(-1LL);
        return X;
    }

} // namespace papi_sde
