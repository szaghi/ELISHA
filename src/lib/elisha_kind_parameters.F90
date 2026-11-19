!< ELISHA, the working real kind of the solvers.
module elisha_kind_parameters
!< ELISHA, the working real kind of the solvers.
!<
!< Every solver computes in PENF's R_P, which is R8P: the solvers stop at relative residuals of 1e-10 to 1e-12 and the
!< self-gravity energy exchange needs the solution map symmetric to round-off, neither of which single precision
!< reaches; the FFT bindings (FFTW `fftw_*`, cuFFT D2Z/Z2D) are the double-precision ones. A single-precision build
!< would need the `fftwf_*` and R2C/C2R bindings and epsilon-scaled tolerances: it is refused until a caller needs
!< it. Mixed precision, if it comes, is a second kind for the multigrid preconditioner, not a global switch.
!<
!< A compile-time check ties the build to PENF's R_P: a build whose PENF R_P is not R8P stops with a division by zero
!< in a constant expression.
use penf, only : R_P, R8P
implicit none
private
public :: R_P

#if defined _R_P_IS_R4P
#error "ELISHA: R4P is not supported (FFT bindings and tolerances are double precision; no caller needs it)"
#elif defined _R_P_IS_R16P
#error "ELISHA: R16P is not supported (no FFT library binding, no caller)"
#else
integer, parameter :: KIND_CHECK = 1/merge(1, 0, R_P == R8P) !< Stops the build if PENF's R_P is not R8P.
#endif
endmodule elisha_kind_parameters
