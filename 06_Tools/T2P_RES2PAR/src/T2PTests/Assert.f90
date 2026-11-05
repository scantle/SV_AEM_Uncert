module assert
  
  ! Roughly based off of and inspired by funit: https://github.com/Goddard-Fortran-Ecosystem/pFUnit/blob/main/src/funit/asserts/AssertBasic.F90
  
  contains
  
  subroutine assert_true(condition, message, errors)
    implicit none
    logical, intent(in)          :: condition
    character(len=*), intent(in) :: message
    integer, intent(inout)       :: errors

    if (.not. condition) then
      write(*,*) message
      errors = errors + 1
    end if
  
  end subroutine assert_true
  
end module assert