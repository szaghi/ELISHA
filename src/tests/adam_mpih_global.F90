!< Test shim: stands in for ADAM's `adam_mpih_global`, just enough for the verbatim `adam_poisson_fft_object`.
module adam_mpih_global
!< Test shim: stands in for ADAM's `adam_mpih_global`, just enough for the verbatim `adam_poisson_fft_object`.
!<
!< Not ADAM's object: the global `mpih` with the members `adam_poisson_fft_object` (ADAM 669d0bdd) reads, and nothing
!< else. The solver reads, through `mpih%`:
!< - `myrank`: this rank in `MPI_COMM_WORLD`;
!< - `myrankstr`: a rank prefix of its warnings;
!< - `procs_number`: the ranks of `MPI_COMM_WORLD`;
!< - `error_stop(msg)`: its fatal path (here: print the message and `error stop`; test code, never the library's).
!< `initialize` fills the rank and the size from `MPI_COMM_WORLD`; the caller has called `MPI_Init` first. ADAM's solver
!< needs MPI: built with `-DELISHA_NO_MPI` the module is empty. Not part of the library: never used by `src/lib`.
#ifndef ELISHA_NO_MPI
use :: mpi
use :: penf, only : I4P
implicit none
private
public :: mpih

type :: mpih_object
   !< ADAM's MPI handler, the members the FFT solver reads.
   integer(I4P)              :: myrank=0_I4P       !< This rank in MPI_COMM_WORLD.
   integer(I4P)              :: procs_number=1_I4P !< Ranks of MPI_COMM_WORLD.
   character(:), allocatable :: myrankstr          !< Rank prefix of messages.
   contains
      procedure, pass(self) :: initialize !< Fill the rank and the size from MPI_COMM_WORLD.
      procedure, pass(self) :: error_stop !< Print a message and stop.
endtype mpih_object

type(mpih_object) :: mpih !< The global handler.

contains
   subroutine initialize(self)
   !< Fill the rank and the size from MPI_COMM_WORLD (MPI initialised by the caller).
   class(mpih_object), intent(inout) :: self !< The handler.
   integer                           :: r, n, ierr !< MPI rank, size, error status.
   character(16)                     :: w    !< Buffer.

   call MPI_COMM_RANK(MPI_COMM_WORLD, r, ierr)
   call MPI_COMM_SIZE(MPI_COMM_WORLD, n, ierr)
   self%myrank = int(r, I4P) ; self%procs_number = int(n, I4P)
   write(w, '(A,I0,A)') 'rank ', r, ': '
   self%myrankstr = trim(w)//' '
   endsubroutine initialize

   subroutine error_stop(self, msg)
   !< Print a message and stop (test code).
   class(mpih_object), intent(in) :: self !< The handler.
   character(*),       intent(in) :: msg  !< Message.

   if (allocated(self%myrankstr)) then
      print '(A)', self%myrankstr//msg
   else
      print '(A)', msg
   endif
   error stop 1
   endsubroutine error_stop
#endif
endmodule adam_mpih_global
