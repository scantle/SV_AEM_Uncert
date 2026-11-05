module tools
  
    real(8), parameter  :: pi = 4.D0*DATAN(1.D0)  ! Used by lognormal dist functions
  
  contains
!-----------------------------------------------------------------------------!
  subroutine loc2glo(locx, locy, xoff, yoff, rot, glox, gloy)
    implicit none

    real    ,intent(in)        :: locx, locy, xoff, yoff, rot
    real    ,intent(out)       :: glox, gloy
    real    , parameter        :: DEG2RAD = 3.141592654/180.0

    glox = xoff + locx * cos(rot*DEG2RAD) - locy * sin(rot*DEG2RAD)
    gloy = yoff + locy * cos(rot*DEG2RAD) + locx * sin(rot*DEG2RAD)

  end subroutine loc2glo
!-----------------------------------------------------------------------------!

!-----------------------------------------------------------------------------!
  subroutine glo2loc(glox, gloy, xoff, yoff, rot, locx, locy)
    implicit none

    real    ,intent(in)        :: glox, gloy, xoff, yoff, rot
    real    ,intent(out)       :: locx, locy
    real    , parameter        :: DEG2RAD = 3.141592654/180.0

    locx = (glox-xoff) * cos(rot*DEG2RAD) + (gloy-yoff) * sin(rot*DEG2RAD)
    locy = (gloy-yoff) * cos(rot*DEG2RAD) - (glox-xoff) * sin(rot*DEG2RAD)

  end subroutine glo2loc
!-----------------------------------------------------------------------------!
  
!-----------------------------------------------------------------------------!
subroutine expand_int_array(array, new_len, init)
    implicit none
    integer, pointer, intent(inout) :: array(:)
    integer, intent(in)             :: new_len
    integer, intent(in), optional   :: init
    integer, pointer                :: temp(:)

    allocate(temp(new_len))
    if (present(init)) temp = init
    temp(1:size(array)) = array
    deallocate(array)
    nullify(array)
    array => temp
end subroutine expand_int_array
  
!-----------------------------------------------------------------------------!
  
!-----------------------------------------------------------------------------!
  subroutine random_seed_initialize (key)
    !*****************************************************************************
    !
    !! random_seed_initialize() initializes the FORTRAN90 random number generator.
    !
    !  Discussion:
    !
    !    This is the stupidest, most awkward procedure I have seen!
    !
    !  Modified:
    !
    !    27 October 2021
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Input:
    !
    !    integer KEY: an initial seed for the random number generator.
    !
    implicit none

    integer key
    integer, allocatable :: seed(:)
    integer seed_size

    if (key<=0) key = huge(seed_size) / 17
    call random_seed ( size = seed_size )
    allocate ( seed(seed_size) )
    seed(1:seed_size) = key
    call random_seed ( put = seed )
    deallocate ( seed )

    return
  end subroutine random_seed_initialize
!-----------------------------------------------------------------------------!

!-----------------------------------------------------------------------------!
function pinpol(px, py, polyx, polyy, npoly) result(inpoly)
  implicit none
  !---------------------------------------------------------------------------!
  !                                 PINPOL
  !                               *********
  !                 Checks if a point is inside a polygon.
  !      "An improved version of the algorithm of Nordbeck and Rydstedt"
  !
  ! Arguments:
  !   -- px is the x-coordinate of the point to be tested
  !   -- py is the y-coordinate of the point to be tested
  !   -- polyx is a vector of nodal x-coordinates in counter-clockwise order
  !   -- polyy is a vector of nodal y-coordinates in counter-clockwise order
  !   -- npoly is the number of polygon verticies
  !
  ! Requires:
  !   -- Function tridet to calculate triangle determinant
  !
  ! Returns:
  !   -- `inpoly` is the square distance from the point to the nearest point 
  !      on the polygon. The sign of `inpoly` indicates the point's position 
  !      relative to the polygon:
  !      - Exact Zero (0.0): The point lies exactly on the boundary of the 
  !        polygon.
  !      - Positive Value: The point is inside the polygon.
  !      - Negative Value: The point is outside the polygon.
  !
  ! Adapted from Sloan, S. W., 1985 A point-in-polygon program, Adv. Eng.
  !                  Software, Vol 7., No. 1
  !
  !               Implemented in f90 by Leland Scantlebury
  !
  ! Copyright 2018-2025 Leland Scantlebury
  !---------------------------------------------------------------------------!

  integer, intent(in)      :: npoly
  integer                  :: i, next, prev, j
  real    , intent(in)     :: px, py, polyx(npoly), polyy(npoly)
  real                     :: x1, y1, x21, y21, x1p, y1p, t, d, dx, dy, area
  real    , parameter      :: smalld = 1e-6
  logical                  :: snear
  real                     :: inpoly

  inpoly = HUGE(1.0)

  ! Loop over each side defining the polygon
  do i=1, npoly
    ! Side coordinates, length, distance from x,y
    next = i + 1
    if (i == npoly) next = 1
    x1 = polyx(i)
    y1 = polyy(i)
    x21 = polyx(next)-x1
    y21 = polyy(next)-y1
    x1p = x1-px
    y1p = y1-py

    ! Find where normal of px,py intersects infinite line
    t = -(x1p*x21 + y1p*y21)/(x21**2 + y21**2)
    if (t < 0.0d0) then
      ! Normal does not intersect side, point is closer to (x1, y1)
      ! Compute square distance to vertex
      d = x1p**2 + y1p**2
      if (d < inpoly) then
        ! Smallest distance yet
        snear = .false.
        inpoly = d
        j = i
      end if
    else if (t < 1.0d0) then
      ! Normal intersects the side
      dx = x1p + t * x21
      dy = y1p + t * y21
      d = dx**2 + dy**2
      if (d < inpoly) then
        ! Smallest distance yet
        snear = .true.
        inpoly = d
        j = i
      end if
    else
      ! Point is closer to the next vertex, continue on to next side
      cycle
    end if
  end do

  if (inpoly < smalld) then
    ! Point lies on the side of the polygon
    inpoly = 0.0d0
  else
    next = j + 1
    prev = j - 1
    if (j == 1) then
      prev = npoly
    else if (j == npoly) then
      next = 1
    else
      continue
    end if
    if (snear) then
      ! Point is closer to side. Use determinant to determine if the triangle
      ! formed by the vertices and point is positive or negative. Positive indicates
      ! the point is inside (form a counter-clockwise triangle)
      area = tridet(polyx(j),polyy(j),polyx(next),polyy(next),px,py)
      inpoly = sign(inpoly, area)
    else
      ! Point is closer to node. Check if nearest vertex is concave
      ! If concave, point is inside the polygon
      !if (j == 1) j = npoly + 1
      area = tridet(polyx(next),polyy(next),polyx(j),polyy(j),polyx(prev),polyy(prev))
      inpoly = sign(inpoly, area)
    end if
  end if

  return
  end function pinpol

!-----------------------------------------------------------------------------!

function tridet(x1, y1, x2, y2, x3, y3) result(det)
  implicit none
!-----------------------------------------------------------------------------!
! Computes twice the area of the triangle defined by coordinates
! (x1,y1) (x2,y2) (x3,y3) using determinate formula
!
! If the area is positive, the points are counter-clockwise
! If the area is negative, the points are clockwise
! If the area is zero, two or more of the points are co-located or all three
!    points are collinear
!
! Useful for determining what side of a line a point (x3, y3) is on.
!-----------------------------------------------------------------------------!
  real    , intent(in)     :: x1, y1, x2, y2, x3, y3
  real                     :: det

  det = (x1-x3)*(y2-y3)-(x2-x3)*(y1-y3)

  return
  end function tridet
  
!-----------------------------------------------------------------------------!
  
function indices_from_mask(mask) result(idx_array)
    implicit none
    ! Returns array of length mask==.true. with the indices of true values from
    ! 1:length(mask) - can be used to subset an array with only the true values
    logical, intent(in)     :: mask(:)
    integer, allocatable    :: idx_array(:)
    integer                 :: count_true, i, idx_counter

    allocate(idx_array(count(mask)))

    ! Fill the result array with indices where mask is true
    idx_counter = 0
    do i = 1, size(mask)
      if (mask(i)) then
        idx_counter = idx_counter + 1
        idx_array(idx_counter) = i
      end if
    end do
    
    return
end function indices_from_mask

!-----------------------------------------------------------------------------!
subroutine gaussian_quadrature(ngq, ndim, blocksize, gqdelxyz, gqweight)
  ! Thanks MOU !
  implicit none
  integer,intent(in) :: ngq, ndim
  real,intent(in)    :: blocksize(ndim)
  real,intent(inout) :: gqdelxyz(ndim,ngq), gqweight(ngq)
  integer            :: i, j, k, ii(3, 64)
  real, parameter :: frac(4)=[0.4305681558,0.1699905218,-0.1699905218,-0.4305681558] ! see Palmer, 1993
  real, parameter :: wgq4(4)=[0.1739274226,0.3260725774, 0.3260725774, 0.1739274226] ! see Palmer, 1993

  do k=1, 4
    do j=1, 4
      do i=1, 4
        ii(1,i+(j-1)*4+(k-1)*16) = i
        ii(2,i+(j-1)*4+(k-1)*16) = j
        ii(3,i+(j-1)*4+(k-1)*16) = k
      end do
    end do
  end do
  gqweight = 1.0
  do i=1, ngq
    do k = 1, ndim
      gqdelxyz(k, i) = frac(ii(k,i)) * blocksize(k)
      gqweight(i) = gqweight(i) *  wgq4(ii(k,i))
    end do
  end do

end subroutine gaussian_quadrature
  
!-----------------------------------------------------------------------------!
! LOGNORMAL DISTRIBUTION ROUTINES
!-----------------------------------------------------------------------------!
! Obtained from https://people.sc.fsu.edu/~jburkardt/f_src/log_normal/log_normal.f90
! On 3/14/2023
!-----------------------------------------------------------------------------!
  
  function lognorm_mu(norm_mu, norm_sigma) result(mu)
    implicit none
    real(kind=8), intent(in) :: norm_mu, norm_sigma
    real(kind=8) :: mu
  
    mu = log(norm_mu**2 / sqrt(norm_sigma**2 + norm_mu**2))
  
  end function lognorm_mu
  
subroutine log_normal_cdf ( x, mu, sigma, cdf )

!*****************************************************************************80
!
!! log_normal_cdf() evaluates the Log Normal CDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    12 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, the argument of the PDF.
!    0.0 < X.
!
!    Input, real ( kind = rk ) MU, SIGMA, the parameters of the PDF.
!    0.0 < SIGMA.
!
!    Output, real ( kind = rk ) CDF, the value of the CDF.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cdf
  real ( kind = rk ) logx
  real ( kind = rk ) mu
  real ( kind = rk ) sigma
  real ( kind = rk ) x

  if ( x <= 0.0D+00 ) then

    cdf = 0.0D+00

  else

    logx = log ( x )

    call normal_cdf ( logx, lognorm_mu(mu, sigma), sigma, cdf ) ! Modified by LS b/c to match scipy input/output

  end if

  return
end

subroutine log_normal_pdf(x, mu, sigma, pdf)
!*****************************************************************************80
!
!! LOG_NORMAL_PDF evaluates the Log Normal PDF.
!
!  Discussion:
!
!    PDF(A,B;X)
!      = exp ( - 0.5 * ( ( log ( X ) - MU ) / SIGMA )^2 )
!        / ( SIGMA * X * sqrt ( 2 * PI ) )
!
!    The Log Normal PDF is also known as the Cobb-Douglas PDF,
!    and as the Antilog_normal PDF.
!
!    The Log Normal PDF describes a variable X whose logarithm
!    is normally distributed.
!
!    The special case MU = 0, SIGMA = 1 is known as Gilbrat's PDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, the argument of the PDF.
!    0.0 < X
!
!    Input, real ( kind = rk ) MU, SIGMA, the parameters of the PDF.
!    0.0 < SIGMA.
!
!    Output, real ( kind = rk ) PDF, the value of the PDF.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) mu
  real ( kind = rk ) pdf

  real ( kind = rk ) sigma
  real ( kind = rk ) x

  if ( x <= 0.0D+00 ) then
    pdf = 0.0D+00
  else
    pdf = exp ( - 0.5D+00 * ( ( log ( x ) - mu ) / sigma ) ** 2 ) &
      / ( sigma * x * sqrt ( 2.0D+00 * pi ) )
  end if

  return
end

subroutine normal_cdf ( x, mu, sigma, cdf )

!*****************************************************************************80
!
!! NORMAL_CDF evaluates the Normal CDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    23 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, the argument of the CDF.
!
!    Input, real ( kind = rk ) MU, SIGMA, the parameters of the PDF.
!    0.0 < SIGMA.
!
!    Output, real ( kind = rk ) CDF, the value of the CDF.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cdf
  real ( kind = rk ) mu
  real ( kind = rk ) sigma
  real ( kind = rk ) x
  real ( kind = rk ) y

  y = ( x - mu ) / sigma

  call normal_01_cdf ( y, cdf )

  return
end

subroutine normal_01_cdf ( x, cdf )

!*****************************************************************************80
!
!! NORMAL_01_CDF evaluates the Normal 01 CDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    AG Adams,
!    Algorithm 39,
!    Areas Under the Normal Curve,
!    Computer Journal,
!    Volume 12, pages 197-198, 1969.
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, the argument of the CDF.
!
!    Output, real ( kind = rk ) CDF, the value of the CDF.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ), parameter :: a1 = 0.398942280444D+00
  real ( kind = rk ), parameter :: a2 = 0.399903438504D+00
  real ( kind = rk ), parameter :: a3 = 5.75885480458D+00
  real ( kind = rk ), parameter :: a4 = 29.8213557808D+00
  real ( kind = rk ), parameter :: a5 = 2.62433121679D+00
  real ( kind = rk ), parameter :: a6 = 48.6959930692D+00
  real ( kind = rk ), parameter :: a7 = 5.92885724438D+00
  real ( kind = rk ), parameter :: b0 = 0.398942280385D+00
  real ( kind = rk ), parameter :: b1 = 3.8052D-08
  real ( kind = rk ), parameter :: b2 = 1.00000615302D+00
  real ( kind = rk ), parameter :: b3 = 3.98064794D-04
  real ( kind = rk ), parameter :: b4 = 1.98615381364D+00
  real ( kind = rk ), parameter :: b5 = 0.151679116635D+00
  real ( kind = rk ), parameter :: b6 = 5.29330324926D+00
  real ( kind = rk ), parameter :: b7 = 4.8385912808D+00
  real ( kind = rk ), parameter :: b8 = 15.1508972451D+00
  real ( kind = rk ), parameter :: b9 = 0.742380924027D+00
  real ( kind = rk ), parameter :: b10 = 30.789933034D+00
  real ( kind = rk ), parameter :: b11 = 3.99019417011D+00
  real ( kind = rk ) cdf
  real ( kind = rk ) q
  real ( kind = rk ) x
  real ( kind = rk ) y
!
!  |X| <= 1.28.
!
  if ( abs ( x ) <= 1.28D+00 ) then

    y = 0.5D+00 * x * x

    q = 0.5D+00 - abs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5 &
      + a6 / ( y + a7 ) ) ) )
!
!  1.28 < |X| <= 12.7
!
  else if ( abs ( x ) <= 12.7D+00 ) then

    y = 0.5D+00 * x * x

    q = exp ( - y ) * b0 / ( abs ( x ) - b1 &
      + b2 / ( abs ( x ) + b3 &
      + b4 / ( abs ( x ) - b5 &
      + b6 / ( abs ( x ) + b7 &
      - b8 / ( abs ( x ) + b9 &
      + b10 / ( abs ( x ) + b11 ) ) ) ) ) )
!
!  12.7 < |X|
!
  else

    q = 0.0D+00

  end if
!
!  Take account of negative X.
!
  if ( x < 0.0D+00 ) then
    cdf = q
  else
    cdf = 1.0D+00 - q
  end if

  return
end

function lognorm_prob(x, delta, mu, sigma, location) result(prob)
  implicit none
  real(kind=8), intent(in) :: x, delta, mu, sigma, location
  real(kind=8) :: right, left, prob, shifted_x
  
  shifted_x = x - location
  call log_normal_cdf(shifted_x + delta, mu, sigma, right)
  call log_normal_cdf(shifted_x - delta, mu, sigma, left)
  
  prob = right - left
  
end function lognorm_prob

!-----------------------------------------------------------------------------!
  
end module tools 