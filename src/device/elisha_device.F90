!< ELISHA, the device entry point: the solvers' OpenACC + cuFFT backends, beside the host library's `elisha`.
module elisha_device
!< ELISHA, the device entry point: the solvers' OpenACC + cuFFT backends, beside the host library's `elisha`.
!<
!< The two device backends, ADAM's FNL twins ported (see their modules): `elisha_spectral_solver_cufft_object`
!< extends `elisha_spectral_solver_object`, `elisha_hierarchical_solver_oac_object` extends
!< `elisha_hierarchical_solver_object` (its root on cuFFT by default). A client uses `elisha` for everything else and
!< this module for the device types; the library modes `*-nvf-oac` build both into one `libelisha.a` (this module is
!< their target, and it uses `elisha`, so the host library is in the archive). The sources live in `src/device/`,
!< outside `src/lib/`: fpm compiles `src/lib/` recursively, and these need `-acc -cudalib=cufft -D_CUFFT`.
use elisha, only : elisha_spectral_solver_object, elisha_hierarchical_solver_object
use elisha_spectral_solver_cufft, only : elisha_spectral_solver_cufft_object
use elisha_hierarchical_solver_oac, only : elisha_hierarchical_solver_oac_object
implicit none
private
public :: elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
endmodule elisha_device
