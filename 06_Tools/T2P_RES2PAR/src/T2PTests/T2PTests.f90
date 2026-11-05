program T2PTests
  use assert
  use DataTypeTests
  implicit none
!-----------------------------------------------------------------------------!
!                             T2PTests
! Unit Testing for Texture2Par within Visual Studio 19
! Simple setup for a (relatively) simple program
! Copyright Leland Scantlebury, S.S. Papadopulos & Assoc.
!
!
! Example from hypothetical "mathman" module
!   call assert_true(multiplier(2.0d0, 3.0d0) == .0d0, &
!                    'Error in function multiplier from mathman module', &
!                    errors)
!-----------------------------------------------------------------------------!
  integer :: errors = 0

  print *, '*********************************'
  print *, '* Texture2Par Unit Test Program *'
  print *, '---------------------------------'
  print *, ' Errors will be printed as they are found'
  print *
  
  ! Data Type Testing
  call test_data_types(errors)

  ! End
  if (errors < 1) then
    write(*,'(/,a)') 'SUCCESS! No errors encountered.'
  else
    write(*,'(/,a,i3,a)') 'FAILURE!', errors, ' errors encountered.'
  end if

end program T2PTests

