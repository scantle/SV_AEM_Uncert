module m_pilotpoints
  use m_datasets, only: t_dataset
  use m_global, only: NODATA
  use m_grid, only: t_grid, create_grid_object
  use m_vstringlist
  use m_file_io, only: vstrlist_to_charvec
  implicit none

  ! Module Variables - parameters ("properties" to avoid confusion) position in pilot point `values` arrays are tracked
  ! globally through the module. This avoids excessive checking for where specific arrays are within `values`.
  ! Defaults
  !    1     2   3   4      5   6    7    8    9
  ! Kmin, Kmax, Ss, Sy, Aniso, Kd, KHp, KVp, STp
  
  integer                       :: nproperties
  integer                       :: naquifer_props
  integer                       :: nglobal_props
  integer                       :: naquitard_props
  integer,allocatable           :: global_props(:)
  integer,allocatable           :: aquifer_props(:)
  integer,allocatable           :: aquitard_props(:)
  type(t_vstringlist)           :: global_propnames              ! Names of properties
  type(t_vstringlist)           :: aquifer_propnames             ! Names of properties
  type(t_vstringlist)           :: propnames                     ! All parameters

!-------------------------------------------------------------------------------------------------!
  type, extends(t_dataset) :: t_pilotpoints
    ! For reference: inherited from t_dataset
    !integer,allocatable           :: category(:)      ! Zones, Hydrostratigraphic Units (HSUs)
    !type(t_grid),pointer          :: grid             ! Grid

    integer,pointer                :: pp_id(:)         ! ID value of each pilot point (index matches pilot_points)
    integer,allocatable            :: zonelist(:)      ! List of "categories" assigned to each pilot point (pilot point zone)
    integer,allocatable            :: active(:)        ! "Active" parameters (used for aquitards parameter subsetting)
    real   ,allocatable            :: values(:,:,:)    ! Data values by parameter (point, parameter, class_id)

    ! Counters
    integer                        :: npp              ! Number of pilot points
    integer                        :: nppzones         ! Number of pilot point zones
    integer                        :: nactive          ! Number of active properties

    contains

      procedure,private            :: get_pp_index
      procedure,public             :: init_pilotpoints
      generic                      :: initialize => init_pilotpoints
      procedure,public             :: read_pp_line
      procedure,public             :: write_pp_table
      procedure,public             :: finalize => finalize_pilotpoints

  end type t_pilotpoints
!-------------------------------------------------------------------------------------------------!

  contains

!-------------------------------------------------------------------------------------------------!
! MODULE PROCEDURES
!-------------------------------------------------------------------------------------------------!
  
  subroutine setup_pilotpoint_property_arrays()
    implicit none
    
    ! Aquifer Properties
    call vstrlist_new(aquifer_propnames)
    call vstrlist_append (aquifer_propnames, "Kmin" )
    call vstrlist_append (aquifer_propnames, "Kmax" )
    call vstrlist_append (aquifer_propnames, "Ss" )
    call vstrlist_append (aquifer_propnames, "Sy" )
    call vstrlist_append (aquifer_propnames, "Aniso" )
    call vstrlist_append (aquifer_propnames, "Kd" )
    naquifer_props = vstrlist_length(aquifer_propnames)
    
    ! Global Properties
    call vstrlist_new(global_propnames)
    call vstrlist_append (global_propnames, "KHp" )
    call vstrlist_append (global_propnames, "KVp" )
    call vstrlist_append (global_propnames, "STp" )
    nglobal_props = vstrlist_length(global_propnames)
    
    ! Aquitard Properties (hardcoded)
    ! (Includes relevant global properties, confusingly)
    naquitard_props = 6  ! (/1,2,5,6,7,8/) by default
  
  end subroutine setup_pilotpoint_property_arrays
  
!-------------------------------------------------------------------------------------------------!

  function create_pilotpoint_object(npoints, nclasses, is_aquitard) result(object)
    implicit none

    integer,intent(inout)            :: npoints,nclasses
    class(t_pilotpoints),pointer     :: object
    integer                          :: i
    logical,intent(in)               :: is_aquitard

    allocate(object)

    call object%initialize(npoints)
    
    ! Allocate the module arrays if they haven't been allocated yet
    if (.not. allocated(aquifer_props)) then
      ! Combine lists to get full pilot point parameter list
      propnames = vstrlist_concat(aquifer_propnames, global_propnames)
      nproperties = vstrlist_length(propnames)
    
      ! Create parameter maps
      allocate(global_props  (nglobal_props),  &
               aquifer_props (naquifer_props), &
               aquitard_props(naquitard_props) )
      ! We want some simple arrays to be able to pull out subsets later.
      aquifer_props = [(i, i=1, naquifer_props)]
      global_props  = [(i, i=naquifer_props+1, naquifer_props + nglobal_props)]
      ! Aquitards is trickier, since it is a subset of both. Searching is easiest.
      aquitard_props(1) = vstrlist_search(propnames, 'Kmin')
      aquitard_props(2) = vstrlist_search(propnames, 'Kmax')
      aquitard_props(3) = vstrlist_search(propnames, 'Aniso')
      aquitard_props(4) = vstrlist_search(propnames, 'Kd')
      aquitard_props(5) = vstrlist_search(propnames, 'KHp')
      aquitard_props(6) = vstrlist_search(propnames, 'KVp')
    end if

    ! Have to allocate this seperately, since init cannot take more than npoints (inheritance)
    ! TODO that didn't have to be true
    allocate(object%values(npoints, nproperties, nclasses))
    object%values   = NODATA
    
    ! Setup active array
    if (is_aquitard) then
      object%active = aquitard_props
    else
      object%active = [(i, i=1,nproperties)]  ! all
    end if
    object%nactive = size(object%active)

  end function
  
!-------------------------------------------------------------------------------------------------!
  
  subroutine add_pilotpoint_aquifer_parameter(name, before)
    ! Must be called prior to create_pilotpoint_object / init_pilotpoints
    implicit none
    character(*), intent(in)     :: name
    integer,intent(in),optional  :: before
    integer                      :: i
    
    if (.not. present(before)) then
      call vstrlist_append(aquifer_propnames, name)
    else
      aquifer_propnames = vstrlist_insert(aquifer_propnames, before, name)
    end if
    naquifer_props = vstrlist_length(aquifer_propnames)
  
  end subroutine add_pilotpoint_aquifer_parameter
  
!-------------------------------------------------------------------------------------------------!

!-------------------------------------------------------------------------------------------------!
! LAYER CLASS TYPE-BOUND PROCEDURES
!-------------------------------------------------------------------------------------------------!

  subroutine init_pilotpoints(this, npoints, grid)
    implicit none
    class(t_pilotpoints),intent(inout) :: this
    integer,intent(in)                 :: npoints
    class(t_grid),pointer,optional     :: grid

    allocate(this%category   (npoints))
    allocate(this%grid)

    this%npp            = npoints

    ! Intialize with default values
    this%nppzones        = 1
    this%category        = 1

    if (present(grid)) then
      this%grid => grid
    else
      this%grid => create_grid_object(2, npoints)  ! X,Y coords by n
    end if
  
  end subroutine init_pilotpoints

!-------------------------------------------------------------------------------------------------!

  function get_pp_index(this, pp_id) result(index)
    use m_error_handler, only: error_handler
    implicit none
    class(t_pilotpoints), intent(inout)   :: this
    integer,intent(in)                    :: pp_id
    integer                               :: index
    character(4)                          :: temp

    if (pp_id > this%npp) then
      write(temp, '(i2)') pp_id
      call error_handler(1,"Pilot Point ID greater than number of pilot points locations: " // trim(temp))
    end if

    do index=1, this%npp
      if (this%pp_id(index)==pp_id) exit
    end do

  end function get_pp_index

!-------------------------------------------------------------------------------------------------!

  subroutine read_pp_line(this, id, pp_type, strings, length, class_id)
    use m_vstringlist, only: t_vstringlist, vstrlist_search, vstrlist_index
    use m_file_io, only: item2int, item2dp
    use m_error_handler, only: error_handler

    implicit none
    class(t_pilotpoints),intent(inout) :: this
    character(*),intent(in)            :: id, pp_type
    type(t_vstringlist),intent(in)     :: strings
    integer,intent(in)                 :: length, class_id
    integer                            :: pp_index, i

    ! Read line for pp_id and get internal pilot point index
    pp_index = this%get_pp_index(item2int(strings, 1))
    ! Now we react to type
    select case(pp_type)
      case("AQUIFER")
        if (length < naquifer_props+2) call error_handler(1,"Missing 1 or more aquifer parameter columns for pilot point " // trim(id))
        do i=1, naquifer_props
          this%values(pp_index,aquifer_props(i),class_id) = item2dp(strings, 2+i)
        end do
        ! K Processing & Checking
        if (this%values(pp_index,2,class_id) < this%values(pp_index,1,class_id)) then
          call error_handler(1,"KMax < Kmin for pilot point " // trim(id))
        else
          ! We krige the *difference* to ensure Kmax > Kmin everywhere
          ! Store this value as a difference
          this%values(pp_index,2,class_id) = this%values(pp_index,2,class_id) - this%values(pp_index,1,class_id)
        end if
      case("AQUITARD")
        if (length < naquitard_props-2) call error_handler(1,"Missing 1 or more aquitard parameter columns for pilot point " // trim(id))
        do i=1, naquitard_props-2  ! Not very dynamic...
          this%values(pp_index,aquitard_props(i),class_id) = item2dp(strings, 2+i)
        end do
        ! K Processing & Checking
        if (this%values(pp_index,2,class_id) < this%values(pp_index,1,class_id)) then
          call error_handler(1,"KMax < Kmin for pilot point " // trim(id))
        else
          ! We krige the *difference* to ensure Kmax > Kmin everywhere
          ! Store this value as a difference
          this%values(pp_index,2,class_id) = this%values(pp_index,2,class_id) - this%values(pp_index,1,class_id)
        end if
      case("GLOBAL")
        if (length < nglobal_props+1) call error_handler(1,"Missing 1 or more global parameter columns for pilot point " // trim(id))
        ! Set for ALL classes
        do i=1, nglobal_props
          this%values(pp_index,global_props(i),:) = item2dp(strings, 1+i)
        end do
        !this%values(pp_index,7, :) = item2dp(strings, 2)  ! KHp
        !this%values(pp_index,8, :) = item2dp(strings, 3)  ! KVp
        !this%values(pp_index,9, :) = item2dp(strings, 4)  ! STp
      case("GLOBAL_AQUITARD")
        if (length < nglobal_props) call error_handler(1,"Missing 1 or more global aquitard parameter columns for pilot point " // trim(id))
        ! Set for ALL classes
        do i=1, nglobal_props-1  ! Not very dynamic...
          this%values(pp_index,global_props(i),:) = item2dp(strings, 1+i)
        end do
        !this%values(pp_index,7, :) = item2dp(strings, 2)  ! KHp
        !this%values(pp_index,8, :) = item2dp(strings, 3)  ! KVp
      case DEFAULT
        ! Outer routine prevents us from getting here
        call error_handler(1,"Unknown Pilot Point Type: " // trim(pp_type))
    end select

  end subroutine read_pp_line

!-------------------------------------------------------------------------------------------------!

  subroutine write_pp_table(this)
    use m_global, only: log
    use m_file_io, only: t_file_writer
    implicit none
    class(t_pilotpoints), intent(in)     :: this

    character(len=5), allocatable :: prop_names(:)
    character(len=256) :: header, line
    integer :: i, j, k
    real :: val

    !allocate(prop_names(this%nactive))
    prop_names = vstrlist_to_charvec(propnames, this%active)

    ! Header line
    header = "   ID  Class             X            Y  Zone "
    do k = 1, this%nactive
      write(header(len_trim(header)+1:), '(A13)') trim(prop_names(k))
    end do
    call log%write_line(trim(header))

    ! Loop over classes, then pilot points
    do j = 1, size(this%values, 3)  ! class_id
      do i = 1, this%npp
        write(line, '(I5,2X,I5,2X,F12.2,1X,F12.2,1X,I5)') this%pp_id(i), j, this%grid%coords(1,i), this%grid%coords(2,i), this%category(i)
        do k = 1, this%nactive
          write(line(len_trim(line)+2:), '(ES12.4)') this%values(i, this%active(k), j)
        end do
        call log%write_line(trim(line))
      end do
    end do

  end subroutine write_pp_table

!-------------------------------------------------------------------------------------------------!

  subroutine finalize_pilotpoints(this)
    implicit none
    class(t_pilotpoints),intent(inout) :: this

    if (allocated(this%category)) deallocate(this%category)
    if (allocated(this%values  )) deallocate(this%values  )
    if (allocated(this%zonelist)) deallocate(this%zonelist)
    if (associated(this%pp_id  )) nullify(this%pp_id)
    if (associated(this%grid   )) nullify(this%grid)

  end subroutine finalize_pilotpoints

!-------------------------------------------------------------------------------------------------!
end module m_pilotpoints