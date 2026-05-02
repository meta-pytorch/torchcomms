      program test_F_interface
          use :: ISO_C_BINDING
          implicit none

#include "f90papi.h"

          integer :: i, j, ret_val, handle, modifier, id, vendor_id
          integer :: dev_count, numas, dummy_val
          character(len=PAPI_MAX_STR_LEN) :: vendor_name, cpu_name
          character(len=1) :: dummy_str
          integer :: family, model, stepping, sockets, cores, threads
          integer :: l1i_size, l1d_size, l2u_size, l3u_size
          integer :: l1i_line_sz, l1d_line_sz, l2u_line_sz, l3u_line_sz
          integer :: l1i_line_cnt, l1d_line_cnt, l2u_line_cnt
          integer :: l3u_line_cnt, l1i_cache_ass, l1d_cache_ass
          integer :: l2u_cache_ass, l3u_cache_ass
          integer :: thr_list(256)
          integer :: mem_size, thread_count

          ret_val = PAPI_VER_CURRENT

          call papif_library_init(ret_val)
          if( ret_val .ne. PAPI_VER_CURRENT ) then
              print *,'Error at papif_init', ret_val, '!=', PAPI_VER_CURRENT
              print *,'PAPI_EINVAL', PAPI_EINVAL
              print *,'PAPI_ENOMEM', PAPI_ENOMEM
              print *,'PAPI_ECMP', PAPI_ECMP
              print *,'PAPI_ESYS', PAPI_ESYS
              stop
          endif

          modifier = PAPI_DEV_TYPE_ENUM__ALL
          call papif_enum_dev_type(modifier, handle, ret_val)

          do while (ret_val == PAPI_OK)
              call papif_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_PAPI_ID, id, dummy_str, ret_val)
              call papif_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID, vendor_id, dummy_str, ret_val)
              call papif_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__CHAR_NAME, dummy_val, vendor_name, ret_val)
              call papif_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, dev_count, dummy_str, ret_val)

              if (id == PAPI_DEV_TYPE_ID__CPU .and. dev_count > 0) then
                  numas = 1
                  do i = 0, dev_count - 1
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_CHAR_NAME, dummy_val, cpu_name, &
                          ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_FAMILY, family, dummy_str, &
                          ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_MODEL, model, dummy_str, &
                          ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_STEPPING, stepping, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT, sockets, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT, numas, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT, cores, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT, threads, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE, l1i_size, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE, l1d_size, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE, l2u_size, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE, l3u_size, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE, l1i_line_sz, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE, l1d_line_sz, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE, l2u_line_sz, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE, l3u_line_sz, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT, l1i_line_cnt,&
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT, l1d_line_cnt,&
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT, l2u_line_cnt,&
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT, l3u_line_cnt,&
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC, l1i_cache_ass, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC, l1d_cache_ass, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC, l2u_cache_ass, &
                          dummy_str, ret_val)
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC, l3u_cache_ass, &
                          dummy_str, ret_val)

                      print *, 'Vendor             : ', trim(vendor_name), '(', vendor_id, ')'
                      print *, 'Id                 : ', i
                      print *, 'Name               : ', trim(cpu_name)
                      print *, 'CPUID              : ', family, '/', model, '/', stepping
                      print *, 'Sockets            : ', sockets
                      print *, 'Numa regions       : ', numas
                      print *, 'Cores/socket       : ', cores
                      print *, 'Cores/numa         : ', threads / numas
                      print *, 'SMT/core           : ', threads / sockets / cores

                      if (l1i_size > 0) then
                          print *, 'L1i cache          : ', 'size/line size/lines/associativity ', &
                              l1i_size / 1024, l1i_line_sz, l1i_line_cnt, l1i_cache_ass
                          print *, 'L1d cache          : ', 'size/line size/lines/associativity ', &
                              l1d_size / 1024, l1d_line_sz, l1d_line_cnt, l1d_cache_ass
                      endif

                      if (l2u_size > 0) then
                          print *, 'L2 cache           : ', 'size/line size/lines/associativity ', &
                              l2u_size / 1024, l2u_line_sz, l2u_line_cnt, l2u_cache_ass
                      endif

                      if (l3u_size > 0) then
                          print *, 'L3 cache           : ', 'size/line size/lines/associativity ', &
                              l3u_size / 1024, l3u_line_sz, l3u_line_cnt, l3u_cache_ass
                      endif

                  enddo

                  do i = 0, numas - 1
                      call papif_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_NUMA_MEM_SIZE, &
                          mem_size, dummy_str, ret_val)
                  enddo
              endif
              call papif_enum_dev_type(modifier, handle, ret_val)
          end do

          call papif_shutdown()

      end program
