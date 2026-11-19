!< ELISHA test: the L0 math equals ADAM's bitwise (oracle: `adam_flail_l0_reference`, ADAM 669d0bdd verbatim).
program elisha_test_adam_parity_l0
!< ELISHA test: the L0 math equals ADAM's bitwise (oracle: `adam_flail_l0_reference`, ADAM 669d0bdd verbatim).
!<
!< Exhaustive over the inputs a solver passes: smooth sizes k = 1..20000; eigenvalues of every mode of nt = 1..256
!< transform cells, five spacings, every order; D(G) stencils; the extension map of every cell of n = 1..24 cells for
!< every pair of kinds, isolated or not; transform sizes for n = 1..48, ngc = 0..4, every kind pair, every path; the
!< boundary-kind values. Bitwise: reals compared by their bit patterns, so a -0 against a +0 counts.
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, ELISHA_D1_COEF, elisha_axis_eigenvalue, elisha_axis_map, elisha_axis_transform_size, &
                   elisha_dg_stencil, elisha_set_plane, elisha_smooth_size, ELISHA_BC_KINDS
use adam_flail_l0_reference, only : D1_COEF, axis_eigenvalue, axis_map, dg_stencil, set_plane, smooth_size, &
                                    transform_sizes
use adam_flail_parameters_reference, only : FLAIL_BC_PERIODIC, FLAIL_BC_DIRICHLET, FLAIL_BC_NEUMANN, &
                                            FLAIL_BC_DIRICHLET_GHOST, FLAIL_BC_ZERO_GHOSTS, FLAIL_BC_FREE_GHOSTS
implicit none
real(R_P),    parameter :: DXS(5)=[1._R_P, 0.5_R_P, 0.1_R_P, 1._R_P/3._R_P, 2.718281828459045_R_P] !< Spacings.
integer(I4P), parameter :: ORDERS(5)=[0_I4P, 2_I4P, 4_I4P, 6_I4P, 8_I4P] !< Operator orders.
integer(I4P)            :: failures, checks                   !< Failed and total checks.
integer(I4P)            :: k, nt, i, p, n, ngc, a, b, t, m, me, ms, se, ss, iso, pth, nul !< Counters.
integer(I4P)            :: kinds(6), bc(2), bc3(2,3), m3(3)   !< Kinds.
real(R_P)               :: ge, gs                             !< Signs.
real(R_P)               :: fe(4,5,6), fs(4,5,6)               !< Arrays for set_plane.
logical                 :: isolated, pcg, nulls(3)            !< Paths.

failures = 0 ; checks = 0
kinds = [FLAIL_BC_PERIODIC, FLAIL_BC_DIRICHLET, FLAIL_BC_NEUMANN, FLAIL_BC_DIRICHLET_GHOST, FLAIL_BC_ZERO_GHOSTS, &
         FLAIL_BC_FREE_GHOSTS]
call check(all(kinds == ELISHA_BC_KINDS), 'boundary-kind values')
call check(all(bits(reshape(D1_COEF, [16])) == bits(reshape(ELISHA_D1_COEF, [16]))), 'D1 coefficients')
do k=1, 20000
   call check(elisha_smooth_size(k) == smooth_size(k), 'smooth size')
enddo
do i=1, size(DXS)
   do nt=1, 256
      do k=0, nt-1
         do p=1, size(ORDERS)
            call check(bits1(elisha_axis_eigenvalue(k, nt, DXS(i), ORDERS(p))) == &
                       bits1(axis_eigenvalue(k, nt, DXS(i), ORDERS(p))), 'eigenvalue')
         enddo
      enddo
   enddo
enddo
do p=2, 8, 2
   call check(all(bits(elisha_dg_stencil(p)) == bits(dg_stencil(p))), 'D(G) stencil')
enddo
do n=1, 24
   do a=1, 4
      do b=1, 4
         bc = [kinds(a), kinds(b)]
         do iso=0, 1
            isolated = iso == 1
            m = elisha_axis_transform_size(n, 2_I4P, bc, isolated, .false., .false.)
            do t=0, m-1
               call elisha_axis_map(n, m, bc, isolated, t, me, ge)
               call axis_map(n, m, bc, isolated, t, ms, gs)
               call check(me == ms .and. bits1(ge) == bits1(gs), 'extension map')
            enddo
         enddo
      enddo
   enddo
enddo
do n=1, 48
   do ngc=0, 4
      do a=1, 6
         do b=1, 6
            do iso=0, 1
               do pth=0, 1
                  do nul=0, 1
                     bc3 = spread([kinds(a), kinds(b)], 2, 3)
                     isolated = iso == 1 ; pcg = pth == 1 ; nulls = [nul == 1, .false., nul == 1]
                     m3 = transform_sizes([n, n + 1, n + 2], ngc, bc3, isolated, pcg, nulls)
                     call check(elisha_axis_transform_size(n, ngc, bc3(:,1), isolated, pcg, nulls(1)) == m3(1) .and. &
                                elisha_axis_transform_size(n + 1, ngc, bc3(:,2), isolated, pcg, nulls(2)) == m3(2) .and. &
                                elisha_axis_transform_size(n + 2, ngc, bc3(:,3), isolated, pcg, nulls(3)) == m3(3), &
                                'transform sizes')
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
enddo
fe = reshape([(real(i, R_P) / 7._R_P, i=1, size(fe))], shape(fe)) ; fs = fe
do t=1, 3
   call elisha_set_plane(fe, t, 0_I4P, 2_I4P, -1._R_P) ; call set_plane(fs, t, 0_I4P, 2_I4P, -1._R_P)
   call elisha_set_plane(fe, t, 1_I4P, 0_I4P, 0._R_P)  ; call set_plane(fs, t, 1_I4P, 0_I4P, 0._R_P)
enddo
call check(all(bits(reshape(fe, [size(fe)])) == bits(reshape(fs, [size(fs)]))), 'set_plane')
print '(A,I0,A,I0,A)', 'elisha_test_adam_parity_l0: ', checks - failures, '/', checks, ' bitwise'
if (failures > 0) error stop 1
print '(A)', 'elisha_test_adam_parity_l0: ok'
contains
   subroutine check(ok, what)
   !< Count a check, report its first failures.
   logical,      intent(in) :: ok   !< Passed.
   character(*), intent(in) :: what !< What.

   checks = checks + 1
   if (ok) return
   failures = failures + 1
   if (failures <= 10) print '(A)', '  FAIL '//what
   endsubroutine check

   elemental function bits1(x) result(b_)
   !< The bit pattern of a real.
   real(R_P), intent(in) :: x  !< Value.
   integer(int64)        :: b_ !< Bits.

   b_ = transfer(x, b_)
   endfunction bits1

   pure function bits(x) result(b_)
   !< The bit patterns of a real array.
   real(R_P), intent(in) :: x(:)        !< Values.
   integer(int64)        :: b_(size(x)) !< Bits.

   b_ = bits1(x)
   endfunction bits
endprogram elisha_test_adam_parity_l0
