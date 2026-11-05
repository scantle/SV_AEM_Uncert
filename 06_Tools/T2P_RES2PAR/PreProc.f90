module m_preproc
  use m_global, only: NODATA
  use m_vstringlist
  use m_vstring, only: t_vstring, vstring_toupper
  use m_file_io
  use m_error_handler, only: error_handler,warning_handler,warnings
  use m_flow_model, only: t_flow_model
  use m_datasets, only: t_layerdata
!-------------------------------------------------------------------------------------------------!
! Preprocessor module: convert non-texture data to a texture *after* interpolation to model grid
! Available preprocessors:
!  - RES2TEXTURE
!-------------------------------------------------------------------------------------------------!
  implicit none
  
  type :: t_preprocessor  ! abstract base class
    character(100)                   :: filename, mult_filename
    integer                          :: nclasses     ! Number of *preprocessor* classes (usually 1)
    integer                          :: ntexclasses  ! Texture datasets (in T2P)
    type(t_vstringlist)              :: class_names  ! *preprocessor* data class names
    class(t_flow_model),pointer      :: fmodel      => null()
    real,allocatable                 :: mult(:,:)    ! multiplier arrays
    
    contains
      procedure,private              :: initialize
      procedure,private              :: apply_mult
      procedure,public               :: read_input_files
      procedure,public               :: to_texture
      procedure,public               :: get_class_id
      procedure,public               :: get_dataset_idx_list
      procedure,public               :: get_dataval_idx_list
  end type
  
  type,  extends(t_preprocessor) :: t_res2texture
    real,allocatable                 :: shapes(:,:,:)    ! lognormal "shape" parameter (~mean)
    real,allocatable                 :: scales(:,:,:)    ! lognormal "scale" parameter (std)
    
    contains
      procedure,public               :: read_input_files => read_input_files_r2t
      procedure,public               :: to_texture       => to_texture_r2t
      procedure,private              :: calc_normalized_likelihoods
  end type
!-------------------------------------------------------------------------------------------------!
  contains
!-------------------------------------------------------------------------------------------------!
! MODULE PROCEDURES
!-------------------------------------------------------------------------------------------------!
  
  function create_preproc_object(preproc_mode_char, ntexclasses, fmodel) result(object)
    implicit none

    character(*),intent(in)                :: preproc_mode_char
    integer, intent(in)                    :: ntexclasses
    class(t_preprocessor),pointer          :: object
    class(t_flow_model),pointer,intent(in) :: fmodel

    select case(preproc_mode_char)
      case ("RES2TEXTURE")
        allocate(t_res2texture::object)
      case DEFAULT
        call error_handler(1,opt_msg="Unknown Preprocessor Mode Name: " // trim(preproc_mode_char))
    end select

    ! Initialize new object
    call object%initialize(ntexclasses, fmodel)

  end function create_preproc_object
  
!-------------------------------------------------------------------------------------------------!
! BASE CLASS TYPE-BOUND PROCEDURES
!-------------------------------------------------------------------------------------------------!
  
  subroutine initialize(this, ntexclasses, fmodel)
    implicit none
    class(t_preprocessor), intent(inout)   :: this
    integer, intent(in)                    :: ntexclasses
    class(t_flow_model),pointer,intent(in) :: fmodel
    
    call vstrlist_new(this%class_names)
    this%nclasses    = 0
    this%ntexclasses = ntexclasses
    this%filename = ''
    this%mult_filename = ''
    this%fmodel => fmodel
  
  end subroutine initialize

!-------------------------------------------------------------------------------------------------!
  
  subroutine read_input_files(this)
    implicit none
    class(t_preprocessor), intent(inout) :: this
    type(t_file_reader), pointer         :: reader
    integer                              :: k, i, inode, lay
    real                                 :: mult
    
    ! by default, just reads in the multiplier file, if set
    if (len_trim(this%mult_filename) > 0) then
      ! allocate
      allocate(this%mult(maxval(this%fmodel%nnodes),this%fmodel%nlayers))
      this%mult = 0
      ! read
      reader => open_file_reader(this%mult_filename)
      read(reader%unit, *)     ! header
      do k=1, this%fmodel%nlayers
        do i=1, this%fmodel%nnodes(k)
          read(reader%unit, *) inode, lay, mult
          this%mult(inode, lay) = mult
        end do
      end do
    end if
    
  end subroutine read_input_files
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine apply_mult(this, layerdata, class_idx)
    ! Multipliers are on variance of dataset, added to data, not on data itself
    implicit none
    class(t_preprocessor), intent(inout) :: this
    type(t_layerdata), intent(inout) :: layerdata(:)  ! (nlayers)
    integer,intent(in)               :: class_idx
    integer                          :: i, k
    
    do k=1, this%fmodel%nlayers
      do i=1, this%fmodel%nnodes(k)
        layerdata(k)%values(class_idx, i) = layerdata(k)%values(class_idx, i) + sqrt(layerdata(k)%variance(class_idx, i)) * this%mult(i,k)
      end do
    end do
  
  end subroutine apply_mult
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine to_texture(this, layerdata)
    implicit none
    class(t_preprocessor), intent(inout) :: this
    type(t_layerdata), intent(inout)     :: layerdata(:,:)  ! (nsets, nlayers)
    integer                              :: i
    
    ! Apply multipliers if present
    if (allocated(this%mult)) then
      do i=1, this%nclasses
        ! Assumption: only act on primary data
        call this%apply_mult(layerdata(1,:), i)
      end do
    end if
    
  end subroutine to_texture
  
!-------------------------------------------------------------------------------------------------!
! T2P class replicating functions  
!-------------------------------------------------------------------------------------------------!
  
  function get_class_id(this, string, adjust_for_infer) result (id)
    ! Returns zero if not found
    implicit none
    class(t_preprocessor), intent(inout)   :: this
    type(t_vstring),intent(in)    :: string
    integer                       :: id
    logical,optional,intent(in)   :: adjust_for_infer  ! not used, just to keep compatability with t2p%get_class_id
    id = vstrlist_search(this%class_names, vstring_toupper(string), exact=.true.)
    return
  end function get_class_id

!-------------------------------------------------------------------------------------------------!
  
  function get_dataset_idx_list(this,class_ids) result(dataset_idx)
    implicit none
    class(t_preprocessor)   :: this
    integer,intent(in)      :: class_ids(:)
    integer,allocatable     :: dataset_idx(:)

    if (allocated(dataset_idx)) deallocate(dataset_idx)
    allocate(dataset_idx(size(class_ids)))
    dataset_idx(:) = 1    ! Will need to revise if we allow multiple preproc classes
    return
  end function

!-------------------------------------------------------------------------------------------------!

  function get_dataval_idx_list(this, class_ids) result (dataval_idx)
    implicit none
    class(t_preprocessor)   :: this
    integer,intent(in)      :: class_ids(:)
    integer,allocatable     :: dataval_idx(:)
    integer                 :: nactive

    if (allocated(dataval_idx)) deallocate(dataval_idx)
    nactive = 1    ! Will need to revise if we allow multiple preproc classes/cokriging
    dataval_idx = class_ids
    where (dataval_idx>nactive) dataval_idx = 1  ! All covariates use idx=1 for values

  end function get_dataval_idx_list
  
!-------------------------------------------------------------------------------------------------!
! RES2TEXTURE TYPE-BOUND PROCEDURES
!-------------------------------------------------------------------------------------------------!
  
  
  subroutine read_input_files_r2t(this)
    implicit none
    class(t_res2texture), intent(inout) :: this
    type(t_file_reader), pointer         :: reader
    integer                              :: i, ierr, inode, lay
    real,allocatable                     :: scales(:), shapes(:)   ! log mean, std
    
    ! do default by calling super
    call read_input_files(this)
    
    ! allocate reading arrays, scale & shape arrays
    allocate(this%scales(this%ntexclasses,maxval(this%fmodel%nnodes),this%fmodel%nlayers), &
             this%shapes(this%ntexclasses,maxval(this%fmodel%nnodes),this%fmodel%nlayers), &
             scales(this%ntexclasses), &
             shapes(this%ntexclasses)  )
    
    ! read in scale & shape values for each texture, for each cell
    reader => open_file_reader(this%filename)
    read(reader%unit, *, iostat=ierr)     ! header
    do
      read(reader%unit, *, iostat=ierr) inode, lay, scales, shapes
      if (ierr/=0) exit
      this%scales(:,inode,lay) = scales
      this%shapes(:,inode,lay) = shapes
    end do
    
  end subroutine read_input_files_r2t
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine to_texture_r2t(this, layerdata)
    implicit none
    class(t_res2texture), intent(inout) :: this
    type(t_layerdata), intent(inout)    :: layerdata(:,:)  ! (nsets, nlayers) with %values(nclasses, npoints)
    real                                :: texture_lhoods(this%ntexclasses)
    integer                             :: i, k
    
    ! Apply multipliers if present
    if (allocated(this%mult)) then
      do i=1, this%nclasses
        ! Assumption: only act on primary data
        call this%apply_mult(layerdata(1,:), i)
      end do
    end if
    
    ! Now the big transform
    do k=1, this%fmodel%nlayers
      do i=1, size(layerdata(1,k)%values, 2)
        if (layerdata(1,k)%values(1,i) <= NODATA) then
          texture_lhoods = NODATA
        else
          call this%calc_normalized_likelihoods(exp(layerdata(1,k)%values(1,i)), this%shapes(:,i,k), this%scales(:,i,k), texture_lhoods)
        end if
        layerdata(1,k)%values(:,i) = texture_lhoods
      end do
    end do
    
  end subroutine to_texture_r2t
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine calc_normalized_likelihoods(this, value, ln_shape, ln_scale, lhood)
    use m_global, only: NODATA
    use tools, only: lognorm_prob
    implicit none
    class(t_res2texture),intent(inout) :: this
    real, intent(in)                   :: value
    real, intent(in)                   :: ln_shape(this%ntexclasses), ln_scale(this%ntexclasses)
    real, intent(inout)                :: lhood(this%ntexclasses)
    real(8),parameter                  :: delta = 1d0, zero = 0.0d0,cutoff=1d-10
    integer                            :: i
     
    lhood = 0.0
    if (value==NODATA) then
      lhood = NODATA
    else if (value < minval(ln_scale)) then
      lhood(minloc(ln_scale)) = 1.0d0
    else if (value > maxval(ln_scale)) then
      lhood(maxloc(ln_scale)) = 1.0d0
    else
      ! Get texture
      do i=1, this%ntexclasses
        lhood(i) = lognorm_prob(value, delta, ln_scale(i), ln_shape(i), 0.0)
        if (lhood(i) < cutoff) lhood(i) = 0.0
      end do
      if (sum(lhood) <= 0.0) then
        ! Prevent divide by zero
        call error_handler(9, opt_msg="Probabilities <= 0 for all texture classes for point")
      end if
      ! Normalize
      lhood = lhood / sum(lhood)
    end if
  
  end subroutine calc_normalized_likelihoods
  
!-------------------------------------------------------------------------------------------------!
  
end module m_preproc