! MIT License
!
! Copyright (c) 2020, The Regents of the University of California,
! through Lawrence Berkeley National Laboratory (subject to receipt of any
! required approvals from the U.S. Dept. of Energy).  All rights reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

module timemory
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    !
    interface
        !
        subroutine c_timemory_named_init_library(name) &
                bind(C, name="timemory_named_init_library")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_named_init_library
        !
        subroutine c_timemory_finalize_library() &
                bind(C, name="timemory_finalize_library")
        end subroutine c_timemory_finalize_library
        !
        subroutine c_timemory_resume() &
                bind(C, name="timemory_resume")
        end subroutine c_timemory_resume
        !
        subroutine c_timemory_pause() &
                bind(C, name="timemory_pause")
        end subroutine c_timemory_pause
        !
        subroutine c_timemory_set_default(name) &
                bind(C, name="timemory_set_default")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_set_default
        !
        subroutine c_timemory_add_components(name) &
                bind(C, name="timemory_add_components")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_add_components
        !
        subroutine c_timemory_remove_components(name) &
                bind(C, name="timemory_remove_components")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_remove_components
        !
        subroutine c_timemory_push_components(name) &
                bind(C, name="timemory_push_components")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_push_components
        !
        subroutine c_timemory_pop_components() &
                bind(C, name="timemory_pop_components")
        end subroutine c_timemory_pop_components
        !
        function c_timemory_get_begin_record(name) &
                result(idx) &
                bind(C, name="timemory_get_begin_record")
            use iso_c_binding, only : C_CHAR, C_INT64_T
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT64_T) :: idx
        end function c_timemory_get_begin_record
        !
        subroutine c_timemory_end_record(idx) &
                bind(C, name="timemory_end_record")
            use iso_c_binding, only : C_INT64_T
            implicit none
            integer(C_INT64_T), intent(IN) :: idx
        end subroutine c_timemory_end_record
        !
        subroutine c_timemory_push_region(name) &
                bind(C, name="timemory_push_region")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_push_region
        !
        subroutine c_timemory_pop_region(name) &
                bind(C, name="timemory_pop_region")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_timemory_pop_region


    end interface

contains
    !
    subroutine timemory_init_library(name)
        use iso_c_binding, only : C_NULL_CHAR, C_CHAR
        character(len=*) :: name
        character(len=1024) :: farg_name
        character(len=1024, kind=C_CHAR) :: carg_name

        if (LEN(name) == 0) then
            call get_command_argument(0, farg_name)
            carg_name = trim(farg_name)//C_NULL_CHAR
        else
            carg_name = trim(name)//C_NULL_CHAR
        end if
        call c_timemory_named_init_library(carg_name)
    end subroutine timemory_init_library
    !
    subroutine timemory_finalize_library()
        call c_timemory_finalize_library()
    end subroutine timemory_finalize_library
    !
    subroutine timemory_pause()
        call c_timemory_pause()
    end subroutine timemory_pause
    !
    subroutine timemory_resume()
        call c_timemory_resume()
    end subroutine timemory_resume
    !
    subroutine timemory_set_default(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_set_default(trim(name)//C_NULL_CHAR)
    end subroutine timemory_set_default
    !
    subroutine timemory_add_components(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_add_components(trim(name)//C_NULL_CHAR)
    end subroutine timemory_add_components
    !
    subroutine timemory_remove_components(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_remove_components(trim(name)//C_NULL_CHAR)
    end subroutine timemory_remove_components
    !
    subroutine timemory_push_components(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_push_components(trim(name)//C_NULL_CHAR)
    end subroutine timemory_push_components
    !
    subroutine timemory_pop_components()
        call c_timemory_pop_components()
    end subroutine timemory_pop_components
    !
    function timemory_get_begin_record(name) &
        result(idx)
        use iso_c_binding, only : C_NULL_CHAR, C_INT64_T
        character(len=*), intent(IN) :: name
        integer(C_INT64_T) :: idx
        idx = c_timemory_get_begin_record(trim(name)//C_NULL_CHAR)
    end function timemory_get_begin_record
    !
    subroutine timemory_end_record(idx)
        use iso_c_binding, only : C_INT64_T
        integer(C_INT64_T), intent(IN) :: idx
        call c_timemory_end_record(idx)
    end subroutine timemory_end_record
    !
    subroutine timemory_push_region(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_push_region(trim(name)//C_NULL_CHAR)
    end subroutine timemory_push_region
    !
    subroutine timemory_pop_region(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        call c_timemory_pop_region(trim(name)//C_NULL_CHAR)
    end subroutine timemory_pop_region

end module timemory
