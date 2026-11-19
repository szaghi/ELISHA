!< ELISHA test: the L0 spectral math by its meaning (order conditions, symbols, sizes), not against a copy.
program elisha_test_spectral
!< ELISHA test: the L0 spectral math by its meaning (order conditions, symbols, sizes), not against a copy.
!<
!< - D1 of order p: `sum_j 2 j a_j = 1` and `sum_j 2 j^(2m+1) a_j = 0` (m = 1..p/2-1), in exact integers after scaling
!<   by the denominators' LCM (the scaled coefficients must be integers to 1e-12 first).
!< - D(G) stencil: symmetric; its moments `sum_o c_o o^(2k)` are 0, 2, 0, ..., 0 for k = 0, 1, 2, .., p/2 (the second
!<   derivative times h^2, order p), in exact integers; p = 2 is [1/4, 0, -1/2, 0, 1/4].
!< - Eigenvalues: the compact one is the symbol of [1, -2, 1]; the wide one is the symbol of the D(G) stencil,
!<   `sum_o c_o cos(o theta) / dx^2`, to 4 eps of the largest term; exactly 0 at theta = 0 and pi.
!< - Smooth size: against a brute-force factorisation for k = 1..4096.
!< - Transform sizes: every branch.
use elisha, only : I4P, I8P, R_P, ELISHA_D1_COEF, elisha_axis_eigenvalue, elisha_axis_transform_size, &
                   elisha_dg_stencil, elisha_smooth_size, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET,  &
                   ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST
implicit none
integer(I8P), parameter :: LCM(4)=[2_I8P, 12_I8P, 60_I8P, 840_I8P] !< Denominators' LCM of D1 of order 2, 4, 6, 8.
real(R_P),    parameter :: PI=acos(-1._R_P)                       !< Pi.
integer(I4P)            :: failures                               !< Failed checks.
integer(I4P)            :: p, h, j, m, o, k, nt                   !< Counters.
integer(I8P)            :: ai(4), ci(-8:8), s                     !< Scaled coefficients, moment.
real(R_P)               :: c(-8:8), th, sym, lam, big, dx         !< Stencil, phase, symbol, eigenvalue.

failures = 0
! D1 order conditions, exact
do h=1, 4
   p = 2 * h
   do j=1, 4
      if (abs(ELISHA_D1_COEF(j,h) * LCM(h) - nint(ELISHA_D1_COEF(j,h) * LCM(h), I8P)) > 1.e-12_R_P) &
         call fail('D1 order '//itoa(p)//': coefficient not a multiple of 1/LCM')
      ai(j) = nint(ELISHA_D1_COEF(j,h) * LCM(h), I8P)
   enddo
   if (any(ai(h+1:4) /= 0_I8P)) call fail('D1 order '//itoa(p)//': nonzero coefficient beyond j = p/2')
   s = sum([(2_I8P * j * ai(j), j=1, h)])
   if (s /= LCM(h)) call fail('D1 order '//itoa(p)//': sum 2 j a_j /= 1')
   do m=1, h-1
      s = sum([(2_I8P * int(j, I8P)**(2*m+1) * ai(j), j=1, h)])
      if (s /= 0_I8P) call fail('D1 order '//itoa(p)//': moment '//itoa(2*m+1)//' /= 0')
   enddo
enddo
! D(G) stencils: symmetry and moments, exact
do h=1, 4
   p = 2 * h
   c = elisha_dg_stencil(p)
   do o=-8, 8
      if (abs(c(o) * LCM(h)**2 - nint(c(o) * LCM(h)**2, I8P)) > 1.e-9_R_P) &
         call fail('D(G) order '//itoa(p)//': entry not a multiple of 1/LCM^2')
      ci(o) = nint(c(o) * LCM(h)**2, I8P)
   enddo
   if (any(ci(-8:-p-1) /= 0_I8P) .or. any(ci(p+1:8) /= 0_I8P)) call fail('D(G) order '//itoa(p)//': reach beyond p')
   if (any(ci(1:8) /= ci(-1:-8:-1))) call fail('D(G) order '//itoa(p)//': not symmetric')
   if (sum(ci) /= 0_I8P) call fail('D(G) order '//itoa(p)//': does not annihilate constants')
   s = sum([(int(o, I8P)**2 * ci(o), o=-8, 8)])
   if (s /= 2_I8P * LCM(h)**2) call fail('D(G) order '//itoa(p)//': second moment /= 2')
   do k=2, h
      s = sum([(int(o, I8P)**(2*k) * ci(o), o=-8, 8)])
      if (s /= 0_I8P) call fail('D(G) order '//itoa(p)//': moment '//itoa(2*k)//' /= 0')
   enddo
enddo
c = elisha_dg_stencil(2_I4P)
if (any(c(-2:2) /= [0.25_R_P, 0._R_P, -0.5_R_P, 0._R_P, 0.25_R_P])) call fail('D(G) order 2 is not [1/4,0,-1/2,0,1/4]')
! eigenvalues: symbols
dx = 0.37_R_P
do nt=2, 40
   do k=0, nt-1
      th = 2._R_P * PI * k / nt
      lam = elisha_axis_eigenvalue(k, nt, dx, 0_I4P)
      sym = (2._R_P * cos(th) - 2._R_P) / dx**2
      if (abs(lam - sym) > 4._R_P * epsilon(1._R_P) * 4._R_P / dx**2) call fail('compact eigenvalue /= symbol')
      if (k == 0 .and. lam /= 0._R_P) call fail('compact eigenvalue at theta = 0 not exactly 0')
      do h=1, 4
         p = 2 * h
         c = elisha_dg_stencil(p)
         lam = elisha_axis_eigenvalue(k, nt, dx, p)
         sym = sum([(c(o) * cos(o * th), o=-8, 8)]) / dx**2
         big = sum(abs(c)) / dx**2
         if (abs(lam - sym) > 16._R_P * epsilon(1._R_P) * big) call fail('D(G) eigenvalue /= symbol, order '//itoa(p))
         if ((k == 0 .or. 2 * k == nt) .and. lam /= 0._R_P) call fail('D(G) eigenvalue at theta 0/pi not exactly 0')
         if (lam > 0._R_P) call fail('D(G) eigenvalue positive')
      enddo
   enddo
enddo
! smooth sizes against brute force
do k=1, 4096
   if (elisha_smooth_size(k) /= brute_smooth(k)) call fail('smooth size of '//itoa(k))
enddo
! transform sizes
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_PERIODIC, ELISHA_BC_PERIODIC], .false., .false., .false.) /= 10) &
   call fail('transform size: periodic')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET], .false., .false., .false.) /= 20) &
   call fail('transform size: Dirichlet/Dirichlet')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_NEUMANN, ELISHA_BC_NEUMANN], .false., .false., .false.) /= 20) &
   call fail('transform size: Neumann/Neumann')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN], .false., .false., .false.) /= 40) &
   call fail('transform size: Dirichlet/Neumann')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_DIRICHLET_GHOST], .false., .false., &
                               .false.) /= 22) call fail('transform size: ghost-Dirichlet')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET], .false., .true., .false.) /= 10) &
   call fail('transform size: CG path')
if (elisha_axis_transform_size(10, 2, [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET], .false., .false., .true.) /= 10) &
   call fail('transform size: null axis')
if (elisha_axis_transform_size(129, 2, [ELISHA_BC_PERIODIC, ELISHA_BC_PERIODIC], .true., .false., .false.) /= 264) &
   call fail('transform size: isolated (2 (129 + 2) = 262 = 2 x 131 -> 264)')
if (failures > 0) then
   print '(A,I0,A)', 'elisha_test_spectral: ', failures, ' failure(s)'
   error stop 1
endif
print '(A)', 'elisha_test_spectral: ok'
contains
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
   character(16)             :: b  !< Buffer.

   write(b, '(I0)') i
   s_ = trim(b)
   endfunction itoa

   pure function brute_smooth(k_) result(m_)
   !< The smallest even m >= k_ whose prime factors are all at most 13, by trial division over every prime.
   integer(I4P), intent(in) :: k_ !< Least size.
   integer(I4P)             :: m_ !< Size.
   integer(I4P)             :: r_, q_ !< Remainder, divisor.
   logical                  :: ok_ !< Smooth.

   m_ = k_
   do
      if (mod(m_, 2) == 0) then
         r_ = m_ ; q_ = 2 ; ok_ = .true.
         do while (r_ > 1)
            if (mod(r_, q_) == 0) then
               if (q_ > 13) then
                  ok_ = .false.
                  exit
               endif
               r_ = r_ / q_
            else
               q_ = q_ + 1
            endif
         enddo
         if (ok_) return
      endif
      m_ = m_ + 1
   enddo
   endfunction brute_smooth
endprogram elisha_test_spectral
