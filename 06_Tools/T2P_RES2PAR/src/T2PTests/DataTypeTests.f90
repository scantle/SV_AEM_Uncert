module DataTypeTests
  use assert
  use list_module
  use kdtree2_module
  implicit none
  
  contains
  !-----------------------------------------------------------------------------------------------!
  ! Main data type testing routine
  !-----------------------------------------------------------------------------------------------!
  subroutine test_data_types(errors)
    implicit none
    integer, intent(inout)  :: errors
    
    call test_stack_type(errors)
    call test_kdtree2_euclidean(errors)
    call test_kdtree2_anisotropic(errors)
    
  end subroutine test_data_types
  !-----------------------------------------------------------------------------------------------!
  
  !-----------------------------------------------------------------------------------------------!
  ! Test Stack 
  subroutine test_stack_type(errors)
    implicit none
    integer, intent(inout)  :: errors
    integer                 :: int_array(4) = (/ 10, 12, 42, 98 /), kernel
    type(stack)             :: teststack
    type(list_node)         :: corn
    
    ! Test empty
    call assert_true(teststack%size == 0, "Error in empty stack type size from list module", errors)
    
    ! Test adding values
    call teststack%push(int_array(1))
    call assert_true(teststack%size == 1, "Error in incremented stack type size from list module", errors)
    
    call teststack%push(int_array(2))
    call assert_true(teststack%size == 2, "Error in incremented stack type size from list module", errors)
    
    call teststack%push(int_array(3))
    call assert_true(teststack%size == 3, "Error in incremented stack type size from list module", errors)
    
    ! Test removing values
    corn = teststack%pop()  ! pop = corn, get it?
    call assert_true(corn%value==int_array(3), "Error in stack pop 1st return value from list module", errors)
    call assert_true(teststack%size == 2, "Error in reduced stack type size from list module", errors)
    
    corn = teststack%pop()
    call assert_true(corn%value==int_array(2), "Error in stack pop 2rd return value from list module", errors)
    call assert_true(teststack%size == 1, "Error in reduced stack type size from list module", errors)
    
    corn = teststack%pop()
    call assert_true(corn%value==int_array(1), "Error in stack pop 3th return value from list module", errors)
    call assert_true(teststack%size == 0, "Error in emptied stack type size from list module", errors)
  
  end subroutine test_stack_type
  !-----------------------------------------------------------------------------------------------!
  
  !-----------------------------------------------------------------------------------------------!
  ! Test KDtrees - Euclidean
  subroutine test_kdtree2_euclidean(errors)
    implicit none
    integer, intent(inout)  :: errors
    type(kdtree2), pointer :: tree
    real*8, dimension(2, 4) :: set
    real*8, dimension(2) :: query
    real*8               :: expected_distance
    type(kdtree2_result), dimension(3) :: results
    integer       :: success

    ! Setup a simple 2D dataset
    set(:, 1) = [1.0, 2.0]
    set(:, 2) = [3.0, 4.0]
    set(:, 3) = [5.0, 6.0]
    set(:, 4) = [7.0, 8.0]

    ! Create k-d tree without a rotation matrix (using Euclidean distance)
    tree => kdtree2_create(set, success, 2,sort=.true.)

    ! Perform a nearest neighbor search for a query point
    query = [3.1, 4.1]
    call kdtree2_n_nearest(tree, query, 1, results)

    ! Assert nearest neighbor is at index 2 (nearest to [3.0, 4.0])
    call assert_true(results(1)%idx == 2, "Failed Euclidean nearest neighbor test", errors)
    
    ! Assert distance matches expected value (within a tolerance)
    expected_distance = (3.1 - 3.0)**2 + (4.1 - 4.0)**2
    call assert_true(abs(results(1)%dis - expected_distance) < 1.0e-5, "Failed Euclidean nearest neighbor distance test", errors)
    
    ! Perform a nearest neighbor search for a query point
    query = [1, 10]
    call kdtree2_n_nearest(tree, query, 3, results)

    ! Closest
    call assert_true(results(1)%idx == 3, "Failed Euclidean nearest neighbor test", errors)
    expected_distance = (1.0 - 5)**2 + (10 - 6)**2
    call assert_true(abs(results(1)%dis - expected_distance) < 1.0e-5, "Failed Euclidean nearest neighbor distance test", errors)
    ! Third closest
    call assert_true(results(3)%idx == 2.or.results(3)%idx == 4, "Failed Euclidean nearest neighbor test", errors) ! tie
    expected_distance = 40
    call assert_true(abs(results(3)%dis - expected_distance) < 1.0e-5, "Failed Euclidean nearest neighbor distance test", errors)
  
  end subroutine test_kdtree2_euclidean
  !-----------------------------------------------------------------------------------------------!
  
  !-----------------------------------------------------------------------------------------------!
  ! Test KDtrees - Anistropic (added for T2P)
  subroutine test_kdtree2_anisotropic(errors)
    integer, intent(inout) :: errors
    type(kdtree2), pointer :: tree
    real(8), dimension(2, 4) :: set
    real(8), dimension(2) :: query
    real(8), dimension(2, 2) :: rotmat, rotmat_update
    real(8)               :: expected_distance
    type(kdtree2_result), dimension(3) :: results
    integer   :: success

    ! Setup a simple 2D dataset
    set(:, 1) = [1.0, 2.0]
    set(:, 2) = [3.0, 4.0]
    set(:, 3) = [5.0, 6.0]
    set(:, 4) = [7.0, 8.0]

    ! Define a simple rotation matrix (ang1=45 degrees, anis1=0.5)
    rotmat(1, :) = [0.707106781, -1.414213562]
    rotmat(2, :) = [0.707106781, 1.414213562]

    ! Create k-d tree with a rotation matrix (using anisotropic distance)
    tree => kdtree2_create(set, success, 2, sort=.true., rotmat=rotmat)

    ! Perform a nearest neighbor search for a query point
    query = [3.1, 4.1]
    call kdtree2_n_nearest(tree, query, 1, results)
    
    ! Assert nearest neighbor is at index 2 (based on the rotated/anisotropic distance)
    ! Answer is the same as before - boring!
    call assert_true(results(1)%idx == 2, "Failed anisotropic nearest neighbor test", errors)
    expected_distance = 0.05
    call assert_true(abs(results(1)%dis - expected_distance) < 1.0e-5, "Failed anisotropic nearest neighbor distance test", errors)
    
    ! Perform a nearest neighbor search for a query point
    ! Answer will differ from euclidean
    query = [1, 10]
    call kdtree2_n_nearest(tree, query, 3, results)

    ! Assert nearest neighbor is at index 4 (based on the rotated/anisotropic distance)
    call assert_true(results(1)%idx == 4, "Failed anisotropic nearest neighbor test", errors)
    expected_distance = 52.0
    call assert_true(abs(results(1)%dis - expected_distance) < 1.0e-5, "Failed anisotropic nearest neighbor distance test", errors)
    ! Third nearest is 2
    call assert_true(results(3)%idx == 2, "Failed anisotropic nearest neighbor test", errors)
    expected_distance = 148
    call assert_true(abs(results(3)%dis - expected_distance) < 1.0e-5, "Failed anisotropic nearest neighbor distance test", errors)
    
    ! Test when a new rotmat is passed
    rotmat_update(1, :) = [0.707106781, -0.707106781]
    rotmat_update(2, :) = [0.707106781, 0.707106781]
    
    call kdtree2_n_nearest(tree, query, 1, results, rotmat=rotmat_update)
    call assert_true(results(1)%idx == 3, "Failed Euclidean nearest neighbor test", errors)
    expected_distance = (1.0 - 5)**2 + (10 - 6)**2
    call assert_true(abs(results(1)%dis - expected_distance) < 1.0e-5, "Failed Euclidean nearest neighbor distance test", errors)

  end subroutine test_kdtree2_anisotropic
  !-----------------------------------------------------------------------------------------------!
  
end module DataTypeTests