!< ELISHA test: the FFTW 3 link (`-D_FFTW`, `-lfftw3`) by one periodic spectral Poisson solve in 1-D.
program elisha_test_fftw
!< ELISHA test: the FFTW 3 link (`-D_FFTW`, `-lfftw3`) by one periodic spectral Poisson solve in 1-D.
!<
!< `phi` is a zero-mean sum of Fourier modes, `f = L_h phi` its 3-point Laplacian; the solve divides each mode of `f`
!< by the eigenvalue of `L_h`, `lambda_k = (2 cos(2 pi k / n) - 2) / h^2` (the spectral path of ADAM's FLAIL), and must
!< return `phi` to round-off. It proves the binding (`bind(C)` interfaces, r2c/c2r, FFTW_ESTIMATE, unnormalised
!< backward transform) before any solver depends on it.
use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_double, c_double_complex, c_associated
use elisha, only : I4P, R8P
implicit none
#ifdef _FFTW
integer(c_int), parameter :: FFTW_ESTIMATE=64_c_int  !< FFTW planner flag, `fftw3.f03`.
integer(I4P),   parameter :: N=64_I4P                !< Cells.
real(R8P),      parameter :: PI=acos(-1._R8P)        !< Pi.
real(R8P),      parameter :: TOL=1.e-12_R8P          !< Max error relative to max |phi|.
interface
   type(c_ptr) function fftw_plan_dft_r2c_1d(n, in, out, flags) bind(C, name='fftw_plan_dft_r2c_1d')
   !< FFTW: plan a 1-D real-to-complex transform.
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n      !< Size.
   real(c_double)            :: in(*)  !< Input.
   complex(c_double_complex) :: out(*) !< Output.
   integer(c_int), value     :: flags  !< Planner flags.
   endfunction fftw_plan_dft_r2c_1d

   type(c_ptr) function fftw_plan_dft_c2r_1d(n, in, out, flags) bind(C, name='fftw_plan_dft_c2r_1d')
   !< FFTW: plan a 1-D complex-to-real transform.
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n      !< Size.
   complex(c_double_complex) :: in(*)  !< Input.
   real(c_double)            :: out(*) !< Output.
   integer(c_int), value     :: flags  !< Planner flags.
   endfunction fftw_plan_dft_c2r_1d

   subroutine fftw_execute_dft_r2c(p, in, out) bind(C, name='fftw_execute_dft_r2c')
   !< FFTW: execute a real-to-complex plan on the given arrays.
   import :: c_ptr, c_double, c_double_complex
   type(c_ptr), value        :: p      !< Plan.
   real(c_double)            :: in(*)  !< Input.
   complex(c_double_complex) :: out(*) !< Output.
   endsubroutine fftw_execute_dft_r2c

   subroutine fftw_execute_dft_c2r(p, in, out) bind(C, name='fftw_execute_dft_c2r')
   !< FFTW: execute a complex-to-real plan on the given arrays.
   import :: c_ptr, c_double, c_double_complex
   type(c_ptr), value        :: p      !< Plan.
   complex(c_double_complex) :: in(*)  !< Input.
   real(c_double)            :: out(*) !< Output.
   endsubroutine fftw_execute_dft_c2r

   subroutine fftw_destroy_plan(p) bind(C, name='fftw_destroy_plan')
   !< FFTW: destroy a plan.
   import :: c_ptr
   type(c_ptr), value :: p !< Plan.
   endsubroutine fftw_destroy_plan
endinterface
real(R8P)    :: h                      !< Cell spacing.
real(R8P)    :: x                      !< Cell centre.
real(R8P)    :: phi(0:N-1)             !< Exact potential.
real(R8P)    :: f(0:N-1)               !< Source, L_h phi; then the solution.
complex(R8P) :: fh(0:N/2)              !< Transform.
real(R8P)    :: lam                    !< Eigenvalue.
real(R8P)    :: err                    !< Max error relative to max |phi|.
type(c_ptr)  :: p_fwd, p_bwd           !< Plans.
integer(I4P) :: i, k                   !< Counters.

h = 1._R8P / N
do i=0, N-1
   x = (i + 0.5_R8P) * h
   phi(i) = sin(2._R8P * PI * x) + 0.25_R8P * cos(6._R8P * PI * x) - 0.1_R8P * sin(14._R8P * PI * x)
enddo
do i=0, N-1
   f(i) = (phi(modulo(i+1, N)) - 2._R8P * phi(i) + phi(modulo(i-1, N))) / h**2
enddo
p_fwd = fftw_plan_dft_r2c_1d(int(N, c_int), f, fh, FFTW_ESTIMATE)
p_bwd = fftw_plan_dft_c2r_1d(int(N, c_int), fh, f, FFTW_ESTIMATE)
if (.not.(c_associated(p_fwd) .and. c_associated(p_bwd))) error stop 'elisha_test_fftw: FFTW returned no plan'
call fftw_execute_dft_r2c(p_fwd, f, fh)
fh(0) = (0._R8P, 0._R8P) ! zero eigenvalue: the mean
do k=1, N/2
   lam = (2._R8P * cos(2._R8P * PI * k / N) - 2._R8P) / h**2
   fh(k) = fh(k) / lam
enddo
call fftw_execute_dft_c2r(p_bwd, fh, f)
f = f / N ! FFTW's backward transform is unnormalised
call fftw_destroy_plan(p_fwd)
call fftw_destroy_plan(p_bwd)
err = maxval(abs(f - phi)) / maxval(abs(phi))
print '(A,ES10.3)', 'elisha_test_fftw: max relative error ', err
if (err > TOL) error stop 'elisha_test_fftw: the spectral solve misses the exact potential'
print '(A)', 'elisha_test_fftw: ok'
#else
error stop 'elisha_test_fftw: built without -D_FFTW (every test mode links FFTW)'
#endif
endprogram elisha_test_fftw
