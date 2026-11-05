program SFR2PAR
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!  SFR2PAR: A utility to parameterize MODFLOW SFR streambed conductivities.
!
!  Workflow:
!    1. Read "SFR2PAR input file" from command-line argument.
!    2. That file has:
!       Line 1: SFR filename (e.g. "myModel.sfr")
!       Line 2: LPF/UPW filename or "NONE" (e.g. "myModel.upw" or "NONE")
!       Lines 3+: multipliers for segments 1..NSEG
!    3. Copy or read old SFR, parse # of reaches, store layer/row/col for each reach
!       plus original SFR bed K if needed.
!    4. If LPF/UPW != "NONE", read the vertical K array: KV(lay, row, col).
!    5. Create a new SFR output file in which each reach's bed K is:
!          KV(lay, row, col) * multiplier(segment)
!       or if "NONE":
!          (original SFR bed K) * multiplier(segment)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  implicit none

  ! Declarations
  character(len=256)   :: infile
  character(len=256)   :: sfr_file(2), lpf_file, temp
  character(len=500)   :: line
  character(len=4)     :: noneString = 'NONE'

  integer              :: nlay, nrow, ncol, i, id, ierr
  real*8, allocatable  :: mult(:)       ! multipliers for each segment
  real*8               :: max_bk

  ! For reading LPF/UPW:
  real*8, allocatable  :: kvArray(:,:,:)   ! store the vertical K array

  ! For reading/writing SFR:
  integer              :: nrch, nseg
  integer, allocatable :: rch_lay(:), rch_row(:), rch_col(:), rch_seg(:), rch_rch(:)
  real*8, allocatable  :: rch_bk(:), rch_len(:), rch_top(:), rch_slope(:), rch_thk(:)

  ! file units
  integer, parameter   :: iuSFR2PAR = 11
  integer, parameter   :: iuSFR     = 12
  integer, parameter   :: iuNEW     = 13
  integer, parameter   :: iuLPF     = 14

  ! Derived or scratch variables
  real*8               :: baseK, newK
  
  ! Write out flag
write(*,'(a)') '                                                                      '
write(*,'(a)') '   _|_|_|  _|_|_|_|  _|_|_|      _|_|    _|_|_|      _|_|    _|_|_|   '
write(*,'(a)') ' _|        _|        _|    _|  _|    _|  _|    _|  _|    _|  _|    _| '
write(*,'(a)') '   _|_|    _|_|_|    _|_|_|        _|    _|_|_|    _|_|_|_|  _|_|_|   '
write(*,'(a)') '       _|  _|        _|    _|    _|      _|        _|    _|  _|    _| '
write(*,'(a)') ' _|_|_|    _|        _|    _|  _|_|_|_|  _|        _|    _|  _|    _| '
write(*,'(a)') '                                                                      '
write(*,'(a)') '        Written by Leland Scantlebury | leland@scantle.com '
write(*,'(a)')

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! 1. Get SFR2PAR input file from command-line
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (command_argument_count() < 1) then
    write(*,*) 'Missing Required Command Line Argument: [input filename]]'
    stop
  end if

  call get_command_argument(1, infile)

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! 2. Read the SFR2PAR input file
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  write(*,'(2a)') '        Input file: ', trim(infile)
  open(unit=iuSFR2PAR, file=trim(infile), status='old', action='read')
  read(iuSFR2PAR,*) sfr_file(1)       ! SFR filename
  read(iuSFR2PAR,*) sfr_file(2)       ! SFR output filename
  read(iuSFR2PAR,*) lpf_file          ! LPF/UPW filename or "NONE"
  read(iuSFR2PAR,*) nlay, nrow, ncol
  read(iuSFR2PAR,*) temp              ! Max streambed conductivity or "NONE"
  
  ! Handle max_bk
  if (trim(adjustl(temp)) == noneString) then
    max_bk = HUGE(max_bk)
  else
    read(temp, *) max_bk
  end if
  
  ! write to screen
  write(*,'(2a)') 'Read from SFR file: ', trim(sfr_file(1))
  write(*,'(2a)') ' Write to SFR file: ', trim(sfr_file(2))
  write(*,'(2a)') '          LPF file: ', trim(lpf_file)
  write(*,'(a)')
   
  open(unit=iuSFR, file=trim(sfr_file(1)), status='old', action='read')
  
  call read_past_sfr_header(iuSFR)
  backspace(iuSFR)
  
  ! Hopefullyl now we're at Data Set 1c
  read(iuSFR, *) nrch, nseg
  nrch = abs(nrch)
  
  ! Allocate
  allocate(mult(nseg), rch_lay(nrch), rch_row(nrch), rch_col(nrch), rch_seg(nrch), &
           rch_rch(nrch), rch_bk(nrch), rch_len(nrch), rch_top(nrch), rch_slope(nrch), rch_thk(nrch))
  mult(1:) = -999.0
  do i = 1, nseg
    read(iuSFR2PAR,*, iostat=ierr) id, mult(id)
  end do
  close(iuSFR2PAR)

  ! Read Data Set 2 nrch times: KRCH IRCH JRCH ISEG IREACH RCHLEN [STRTOP] [SLOPE] [STRTHICK] [STRHC1] (not implemented [THTS] [THTI] [EPS] [UHC])
  do i = 1, nrch
    read(iuSFR,*) rch_lay(i), rch_row(i), rch_col(i), rch_seg(i), rch_rch(i), rch_len(i), rch_top(i), rch_slope(i), rch_thk(i), rch_bk(i)
  end do

  close(iuSFR)

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! 3. If LPF/UPW != 'NONE', read the vertical K array
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (trim(adjustl(lpf_file)) /= noneString) then
    call read_kv_from_lpf(lpf_file, nlay, nrow, ncol, kvArray)
  end if

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! 4. Read/Write SFR file, rewriting with updated bed K
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  open(unit=iuSFR, file=trim(sfr_file(1)), status='old', action='read')
  open(unit=iuNEW, file=trim(sfr_file(2)), status='replace', action='write')

  call read_past_sfr_header(iuSFR, iuNEW)

  ! Now for the reach definitions:
  do i = 1, nrch
    read(iuSFR,'(a)') line  

    if (mult(rch_seg(i)) >-999) then ! if mult is -999 we don't replace the line
      ! baseK depends on whether we are using LPF/UPW or not
      if (trim(adjustl(lpf_file)) == noneString) then
        baseK = rch_bk(i)  ! from SFR file
      else
        baseK = kvArray(rch_lay(i), rch_row(i), rch_col(i))
      end if

      ! Multiply by segment multiplier
      newK = min(baseK * mult(rch_seg(i)), max_bk)

      ! Now write the updated line to the new SFR file
      ! Keep everything else the same:
      write(iuNEW,'(5I5,5G15.7)')  rch_lay(i), rch_row(i), rch_col(i), rch_seg(i), rch_rch(i), rch_len(i), rch_top(i), rch_slope(i), rch_thk(i), newK
    else
      !write(iuNEW,'(a)') line
      ! Let's be cool and write it consistently
      write(iuNEW,'(5I5,5G15.7)')  rch_lay(i), rch_row(i), rch_col(i), rch_seg(i), rch_rch(i), rch_len(i), rch_top(i), rch_slope(i), rch_thk(i), rch_bk(i)
    end if
  end do

  do
    read(iuSFR,'(a)', iostat=ierr) line
    if (ierr /= 0) exit
    write(iuNEW,'(A)') trim(line)
  end do

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! 5. Close files // Exit out
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  close(iuSFR)
  close(iuNEW)

  print *, 'SFR2PAR: Finished writing updated SFR file => ', trim(sfr_file(2))
  stop
  
  contains

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Subroutines
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
subroutine read_past_sfr_header(unit, w_unit)
  implicit none
  
  integer, intent(in) :: unit
  character(len=256)  :: line
  integer             :: ierr
  integer, intent(in), optional :: w_unit  ! Write unit
  
  do
    read(iuSFR, '(a)', iostat=ierr) line
    if (present(w_unit)) write(w_unit, '(a)') trim(line)
    if (ierr /= 0) then
      write(*,*) "Error during SFR file read - could not identify Data Set 1c line"
      stop
    end if
    line = trim(adjustl(line))
    if (line(1:1)=="#") then
      cycle
    else if (line(1:1)=="T".or.line(1:1)=="R") then
      cycle
    else if (line(1:7)=="OPTIONS") then
      do 
        read(iuSFR, '(a)', iostat=ierr) line
        if (present(w_unit)) write(w_unit, '(a)') trim(line)
        if (ierr /= 0) then
          write(*,*) "Error during SFR file read - could not identify end of OPTIONS"
          stop
        end if
        line = trim(adjustl(line))
        if (line(1:3)=="END") exit
      end do
      cycle
    else
      exit
    end if
  end do

end subroutine read_past_sfr_header
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
subroutine read_kv_from_lpf(lpf_file, nlay, nrow, ncol, kvArray)
    implicit none
    ! Input parameters
    character(len=*), intent(in) :: lpf_file
    integer, intent(in) :: nlay, nrow, ncol

    ! Output parameter
    real(8), allocatable, intent(out) :: kvArray(:,:,:)

    ! Local variables
    integer :: iuLPF, i, j, k, ios, arrayIndex
    character(len=256) :: line
    real(8) :: constMultiplier
    character(len=20) :: formatStr

    ! Allocate the kvArray
    allocate(kvArray(nlay, nrow, ncol))
    kvArray = 0.0  ! Initialize to avoid uninitialized values

    ! File unit number for the LPF file
    iuLPF = 14

    ! Open the LPF file
    open(unit=iuLPF, file=trim(lpf_file), status='old', action='read')

    arrayIndex = 0  ! Reset array index to track position in array sequence
    k = 1

    ! Loop through the LPF file to find Kv data
    do
      ! Read each line as a string to avoid formatting issues
      read(iuLPF, '(A)', iostat=ios) line
      if (ios /= 0) exit  ! Exit on end of file or read error

      ! Skip comment lines
      if (index(line, '#') == 1) cycle

      ! Check if this line is an array header (location, multiplier, format, etc.)
      if (index(line,'INTERNAL') > 0) then
        arrayIndex = arrayIndex + 1
        if (arrayIndex> 4) then
          arrayIndex = 1
          k = k + 1
        end if
      else if (index(line,'EXTERNAL') > 0) then
        write(*,*) 'ERROR - EXTERNAL LPF/UPW arrays not supported'
        stop
      end if
      

      ! When arrayIndex reaches 2, we are at the Kv array (VKA)
      ! TODO handle HANI (CHANI <= 0)
      if (arrayIndex == 2) then
        print *, 'Found Kv array, reading data for Layer', k

        ! Read the Kv array for each layer
        do i = 1, nrow
          read(iuLPF, '(10E12.4)', iostat=ios) ( kvArray(k, i, j), j = 1, ncol )
        end do

      end if
    end do

    ! Close the LPF file
    close(iuLPF)

    ! Ensure Kv array was actually read
    if (arrayIndex < 3) then
        print *, 'Error: Kv array not found in LPF file.'
    end if

end subroutine read_kv_from_lpf
  
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end program SFR2PAR
