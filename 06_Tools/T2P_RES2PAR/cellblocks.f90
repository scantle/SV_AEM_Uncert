module m_cellblocks
  implicit none  
!-------------------------------------------------------------------------------------------------!
! Block Kriging Container
  
  type t_cellblocks
    integer              :: npts = 0
    integer, allocatable :: nblockpts(:)          ! (nodes)
    real,    allocatable :: block_wts(:,:)        ! (npts, nodes)
    real,    allocatable :: block_coords(:,:,:)   ! (dim, npts, nodes)
  end type

!-------------------------------------------------------------------------------------------------!
  
  contains
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine alloc_cellblocks(blck, npts, nnode, dim)
    implicit none
    type(t_cellblocks), intent(inout) :: blck
    integer, intent(in) :: npts, nnode, dim
    blck%npts = npts
    if (allocated(blck%nblockpts))    deallocate(blck%nblockpts)
    if (allocated(blck%block_wts))    deallocate(blck%block_wts)
    if (allocated(blck%block_coords)) deallocate(blck%block_coords)
    allocate(blck%nblockpts(nnode))
    allocate(blck%block_wts(npts, nnode))
    allocate(blck%block_coords(dim, npts, nnode))
  end subroutine
  
!-------------------------------------------------------------------------------------------------!
  
end module m_cellblocks