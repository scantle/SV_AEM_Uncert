module InterpTests
  use assert
  use kdtree2_module
  use m_vario
  !use m_kriging
  implicit none
  
  contains
  !-----------------------------------------------------------------------------------------------!
  ! Main data type testing routine
  !-----------------------------------------------------------------------------------------------!
  subroutine test_interpolators(errors)
    implicit none
    integer, intent(inout)  :: errors
    
    call test_variograms(errors)
    
  end subroutine test_interpolators
  !-----------------------------------------------------------------------------------------------!
  
  subroutine test_variograms(errors)
    implicit none
    integer, intent(inout)  :: errors
    
    
    
  end subroutine test_variograms
  
  !-----------------------------------------------------------------------------------------------!
  
  end module InterpTests