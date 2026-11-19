!< ELISHA test: the extension maps by their meaning, the reflections about each face kind.
program elisha_test_extension
!< ELISHA test: the extension maps by their meaning, the reflections about each face kind.
!<
!< For an axis of n cells and every kind pair, the extended array `e(t) = sg(t) f(src(t))` over the transform cells is
!< checked against the reflection that defines it (indices modulo m, f(j) for j = 0..n-1):
!< - a Dirichlet (Neumann) high face: `e(n + j) = -f(n - 1 - j)` (`+f`), the reflection about the face;
!< - a Dirichlet (Neumann) low face: `e(m - 1 - j) = -f(j)` (`+f`);
!< - ghost-Dirichlet: zero at the ghost centres n and m - 1, `e(n + 1 + j) = -f(n - 1 - j)`, `e(m - 2 - j) = -f(j)`;
!< - isolated: zero padding beyond n.
!< so the periodic 3-point operator on the extension, read at the domain's first and last cells, sees the ghost of
!< each kind (-phi, +phi, 0). Then `elisha_set_plane` against the map along each axis of a 3-D array.
use elisha, only : I4P, R_P, elisha_axis_map, elisha_axis_transform_size, elisha_set_plane, ELISHA_BC_DIRICHLET, &
                   ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_PERIODIC
implicit none
integer(I4P), parameter :: N=7_I4P                      !< Cells.
integer(I4P)            :: failures                     !< Failed checks.
integer(I4P)            :: kinds(2,5)                   !< Kind pairs.
integer(I4P)            :: c, j, m, t, src, d           !< Counters.
real(R_P)               :: f(0:N-1), sg, slo, shi       !< Domain values, sign, face signs.
real(R_P), allocatable  :: e(:), a(:,:,:), b(:,:,:)     !< Extension, 3-D arrays.

failures = 0
f = [(real(3 * j + 1, R_P) / 7._R_P, j=0, N-1)]
kinds(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET]
kinds(:,2) = [ELISHA_BC_NEUMANN, ELISHA_BC_NEUMANN]
kinds(:,3) = [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN]
kinds(:,4) = [ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET]
kinds(:,5) = [ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_DIRICHLET_GHOST]
do c=1, 5
   m = elisha_axis_transform_size(N, 2_I4P, kinds(:,c), .false., .false., .false.)
   call reset(e, m) ; e(0:) = extend(kinds(:,c), m, .false.)
   if (c < 5) then
      slo = merge(-1._R_P, 1._R_P, kinds(1,c) == ELISHA_BC_DIRICHLET)
      shi = merge(-1._R_P, 1._R_P, kinds(2,c) == ELISHA_BC_DIRICHLET)
      do j=0, N-1
         if (e(N + j) /= shi * f(N - 1 - j))     call fail('high-face reflection, pair '//itoa(c))
         if (e(m - 1 - j) /= slo * f(j))         call fail('low-face reflection, pair '//itoa(c))
      enddo
   else
      if (e(N) /= 0._R_P .or. e(m - 1) /= 0._R_P) call fail('ghost-Dirichlet: ghost centres not zero')
      do j=0, N-1
         if (e(N + 1 + j) /= -f(N - 1 - j))      call fail('ghost-Dirichlet: odd about the high ghost centre')
         if (e(m - 2 - j) /= -f(j))              call fail('ghost-Dirichlet: odd about the low ghost centre')
      enddo
   endif
enddo
m = elisha_axis_transform_size(N, 2_I4P, [ELISHA_BC_PERIODIC, ELISHA_BC_PERIODIC], .true., .false., .false.)
call reset(e, m) ; e(0:) = extend([ELISHA_BC_PERIODIC, ELISHA_BC_PERIODIC], m, .true.)
if (any(e(0:N-1) /= f) .or. any(e(N:m-1) /= 0._R_P)) call fail('isolated: not the zero padding')
! set_plane along each axis of a 3-D array, Dirichlet/Neumann (m = 4n)
do d=1, 3
   m = 4 * N
   allocate(a(merge(m, N, d == 1), merge(m, N, d == 2), merge(m, N, d == 3)))
   a = 0._R_P
   select case(d)
   case(1) ; a(1:N,:,:) = spread(spread(f, 2, N), 3, N)
   case(2) ; a(:,1:N,:) = spread(spread(f, 1, N), 3, N)
   case(3) ; a(:,:,1:N) = spread(spread(f, 1, N), 1, N)
   endselect
   b = a
   do t=N, m-1
      call elisha_axis_map(N, m, kinds(:,3), .false., t, src, sg)
      call elisha_set_plane(b, d, t, src, sg)
   enddo
   call reset(e, m) ; e(0:) = extend(kinds(:,3), m, .false.)
   select case(d)
   case(1) ; if (any(b(:,3,5) /= e)) call fail('set_plane along x')
   case(2) ; if (any(b(2,:,6) /= e)) call fail('set_plane along y')
   case(3) ; if (any(b(4,1,:) /= e)) call fail('set_plane along z')
   endselect
   deallocate(a, b)
enddo
if (failures > 0) then
   print '(A,I0,A)', 'elisha_test_extension: ', failures, ' failure(s)'
   error stop 1
endif
print '(A)', 'elisha_test_extension: ok'
contains
   subroutine reset(x, m_)
   !< x allocated as x(0:m_-1) (an assignment from a function result would give it the lower bound 1).
   real(R_P), allocatable, intent(inout) :: x(:) !< Array.
   integer(I4P),           intent(in)    :: m_   !< Size.

   if (allocated(x)) deallocate(x)
   allocate(x(0:m_-1))
   endsubroutine reset

   function extend(bc, m_, isolated) result(e_)
   !< The extension of f over m_ transform cells by the map.
   integer(I4P), intent(in) :: bc(2)     !< Face kinds.
   integer(I4P), intent(in) :: m_        !< Transform cells.
   logical,      intent(in) :: isolated  !< Zero padding.
   real(R_P)                :: e_(0:m_-1) !< Extension.
   integer(I4P)             :: t_, s_    !< Target, source.
   real(R_P)                :: g_        !< Sign.

   do t_=0, m_-1
      call elisha_axis_map(N, m_, bc, isolated, t_, s_, g_)
      if (s_ < 0) then
         if (g_ /= 0._R_P) call fail('a zero cell with a nonzero sign')
         e_(t_) = 0._R_P
      else
         e_(t_) = g_ * f(s_)
      endif
   enddo
   endfunction extend

   subroutine fail(msg)
   !< Report a failed check.
   character(*), intent(in) :: msg !< What failed.

   failures = failures + 1
   print '(A)', '  FAIL '//msg
   endsubroutine fail

   pure function itoa(i) result(s_)
   !< An integer as text.
   integer(I4P), intent(in)  :: i  !< Integer.
   character(:), allocatable :: s_ !< Text.
   character(16)             :: b_ !< Buffer.

   write(b_, '(I0)') i
   s_ = trim(b_)
   endfunction itoa
endprogram elisha_test_extension
