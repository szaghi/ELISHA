!< ELISHA, the communication context: the caller's communicator, duplicated, or one rank without MPI.
module elisha_comm
!< ELISHA, the communication context: the caller's communicator, duplicated, or one rank without MPI.
!<
!< ADAM's FLAIL called every collective on `MPI_COMM_WORLD` (39 sites, audit A1); ELISHA calls them on the communicator
!< its caller gives, duplicated by `init` (`MPI_Comm_dup`), so the library's messages never match the caller's and two
!< solvers can live on two sub-communicators. Every collective of the library goes through this object.
!<
!< Built with `-DELISHA_NO_MPI` there is no MPI at all: `init` ignores the communicator, the rank is 0 of 1, the
!< reductions return their argument and the gathers, scatters and exchanges copy it (one rank: the root's own share).
!< The caller initialises and finalises MPI; `init` before `MPI_Init` returns `ELISHA_ERR_MPI_NOT_INITIALIZED`. MPI
!< handles are default integers (the `mpi` module).
!<
!< The collectives of the spectral solver (ported from ADAM's `adam_poisson_fft_object`) keep ADAM's MPI calls exactly:
!< the caller gives the counts and displacements (in items, as MPI takes them), the datatypes are ADAM's (`MPI_INTEGER`,
!< `MPI_REAL8`, `MPI_DOUBLE_COMPLEX`), the root is rank 0, and the buffers are assumed-size (`x(*)`, MPI's choice
!< buffers: a contiguous array of any rank passes by sequence association, as ADAM passed it to MPI). Only the counts
!< arrays significant at the root (or at no rank, for the exchanges) are read on one rank.
!<
!< The hierarchical solver (ported from ADAM's `adam_poisson_mg_object`) adds the reductions of ADAM's `MPI_REAL8`
!< scalars (`sum_real8`, `max_real8`: in place, the datatype ADAM reduced with, not `sum_r`'s `MPI_DOUBLE_PRECISION`)
!< and the exchange of its cell keys (`alltoallv_i8`, `MPI_INTEGER8`); its gathers and halo exchanges are the
!< spectral solver's.
#ifndef ELISHA_NO_MPI
use mpi
#endif
use penf, only : I4P, I8P
use elisha_kind_parameters, only : R_P
use elisha_status, only : ELISHA_OK, ELISHA_ERR_MPI, ELISHA_ERR_MPI_NOT_INITIALIZED, elisha_set_error
implicit none
private
public :: elisha_comm_object

type :: elisha_comm_object
   !< The communication context of a solver.
   integer :: comm=-1              !< The duplicated communicator (MPI handle; -1 without MPI or before init).
   integer :: rank=0               !< This rank.
   integer :: nranks=1             !< Ranks.
   logical :: is_initialized=.false. !< `init` succeeded.
   contains
      procedure, pass(self) :: init        !< Duplicate the caller's communicator.
      procedure, pass(self) :: destroy     !< Free the duplicate.
      procedure, pass(self) :: min_i4      !< In-place minimum of an integer array over the ranks.
      procedure, pass(self) :: max_i4      !< In-place maximum of an integer array over the ranks.
      procedure, pass(self) :: sum_i8      !< In-place sum of an int64 array over the ranks.
      procedure, pass(self) :: sum_r       !< In-place sum of a real array over the ranks.
      procedure, pass(self) :: allgatherv_i4 !< Every rank's integer array, concatenated in rank order, on every rank.
      ! the spectral solver's collectives: ADAM's MPI calls, caller-given layouts (see the module)
      procedure, pass(self) :: gather_i4            !< Rank 0 gets one integer of every rank (MPI_GATHER).
      procedure, pass(self) :: allgather_i4         !< Every rank gets one integer of every rank (MPI_ALLGATHER).
      procedure, pass(self) :: gatherv_i4           !< Rank 0 gets every rank's integers (MPI_GATHERV).
      procedure, pass(self) :: gatherv_r            !< Rank 0 gets every rank's reals (MPI_GATHERV, MPI_REAL8).
      procedure, pass(self) :: scatterv_r           !< Every rank gets its share of rank 0's reals (MPI_SCATTERV).
      procedure, pass(self) :: allgatherv_layout_i4 !< Every rank gets every rank's integers (MPI_ALLGATHERV).
      procedure, pass(self) :: allgatherv_layout_r  !< Every rank gets every rank's reals (MPI_ALLGATHERV, MPI_REAL8).
      procedure, pass(self) :: bcast_r              !< Rank 0's reals to every rank (MPI_BCAST, MPI_REAL8).
      procedure, pass(self) :: alltoall_i4          !< One integer to and from every rank (MPI_ALLTOALL).
      procedure, pass(self) :: alltoallv_i4         !< Integers to and from every rank (MPI_ALLTOALLV).
      procedure, pass(self) :: alltoallv_r          !< Reals to and from every rank (MPI_ALLTOALLV, MPI_REAL8).
      procedure, pass(self) :: alltoallv_c          !< Complexes to and from every rank (MPI_ALLTOALLV, double complex).
      ! the hierarchical solver's collectives: ADAM's MPI calls (see the module)
      procedure, pass(self) :: sum_real8            !< In-place sum of a real array over the ranks (MPI_REAL8).
      procedure, pass(self) :: max_real8            !< In-place maximum of a real array over the ranks (MPI_REAL8).
      procedure, pass(self) :: alltoallv_i8         !< int64 values to and from every rank (MPI_ALLTOALLV, MPI_INTEGER8).
      procedure, nopass     :: wtime                !< Wall-clock time in seconds (MPI_WTIME; system_clock without MPI).
endtype elisha_comm_object

contains
   subroutine init(self, comm, stat, errmsg)
   !< Duplicate the caller's communicator (ignored without MPI); a second `init` frees the first duplicate.
   class(elisha_comm_object), intent(inout)         :: self   !< Context.
   integer,                   intent(in)            :: comm   !< The caller's communicator.
   integer(I4P),              intent(out)           :: stat   !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
#ifndef ELISHA_NO_MPI
   logical                                          :: ok     !< MPI is initialised.
   integer                                          :: ierr   !< MPI error status.

   if (self%is_initialized) then
      call self%destroy(stat=stat, errmsg=errmsg)
      if (stat /= ELISHA_OK) return
   endif
   call MPI_INITIALIZED(ok, ierr)
   if (.not.ok) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MPI_NOT_INITIALIZED, 'elisha_comm_object%init: call MPI_Init first')
      return
   endif
   call MPI_COMM_DUP(comm, self%comm, ierr)
   if (ierr == MPI_SUCCESS) call MPI_COMM_RANK(self%comm, self%rank, ierr)
   if (ierr == MPI_SUCCESS) call MPI_COMM_SIZE(self%comm, self%nranks, ierr)
   if (ierr /= MPI_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MPI, 'elisha_comm_object%init: MPI_Comm_dup/rank/size failed')
      return
   endif
#else
   if (.false.) self%comm = comm ! without MPI the communicator is ignored
   self%comm = -1 ; self%rank = 0 ; self%nranks = 1
#endif
   self%is_initialized = .true.
   stat = ELISHA_OK
   endsubroutine init

   subroutine destroy(self, stat, errmsg)
   !< Free the duplicate (nothing without MPI).
   class(elisha_comm_object), intent(inout)         :: self   !< Context.
   integer(I4P),              intent(out)           :: stat   !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
#ifndef ELISHA_NO_MPI
   integer                                          :: ierr   !< MPI error status.

   if (self%is_initialized) then
      call MPI_COMM_FREE(self%comm, ierr)
      if (ierr /= MPI_SUCCESS) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_MPI, 'elisha_comm_object%destroy: MPI_Comm_free failed')
         return
      endif
   endif
#endif
   self%comm = -1 ; self%rank = 0 ; self%nranks = 1 ; self%is_initialized = .false.
   stat = ELISHA_OK
   endsubroutine destroy

   subroutine min_i4(self, x, stat)
   !< x <- the elementwise minimum of x over the ranks.
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I4P),              intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_INTEGER, MPI_MIN, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine min_i4

   subroutine max_i4(self, x, stat)
   !< x <- the elementwise maximum of x over the ranks.
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I4P),              intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_INTEGER, MPI_MAX, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine max_i4

   subroutine sum_i8(self, x, stat)
   !< x <- the elementwise sum of x over the ranks.
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I8P),              intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_INTEGER8, MPI_SUM, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine sum_i8

   subroutine sum_r(self, x, stat)
   !< x <- the elementwise sum of x over the ranks (R_P is R8P: MPI_DOUBLE_PRECISION).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   real(R_P),                 intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine sum_r

   subroutine allgatherv_i4(self, x, y, counts, stat)
   !< y <- every rank's x, concatenated in rank order; counts(r+1) the size of rank r's x.
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   integer(I4P),              intent(in)    :: x(:)      !< This rank's values.
   integer(I4P), allocatable, intent(inout) :: y(:)      !< Every rank's values.
   integer(I4P), allocatable, intent(inout) :: counts(:) !< Values per rank [nranks].
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer, allocatable                     :: c(:), dsp(:) !< Counts and displacements (default integers).
   integer                                  :: r, ierr      !< Counter, MPI error status.

   allocate(c(self%nranks), dsp(self%nranks))
   call MPI_ALLGATHER(size(x), 1, MPI_INTEGER, c, 1, MPI_INTEGER, self%comm, ierr)
   if (ierr /= MPI_SUCCESS) then
      stat = ELISHA_ERR_MPI
      return
   endif
   dsp(1) = 0
   do r=2, self%nranks
      dsp(r) = dsp(r-1) + c(r-1)
   enddo
   if (allocated(y)) deallocate(y)
   allocate(y(max(sum(c), 1)))
   call MPI_ALLGATHERV(x, size(x), MPI_INTEGER, y, c, dsp, MPI_INTEGER, self%comm, ierr)
   if (sum(c) == 0) then ! the receive buffer had one dummy entry
      deallocate(y)
      allocate(y(0))
   endif
   counts = int(c, I4P)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   y = x
   counts = [size(x, kind=I4P)]
   stat = ELISHA_OK
#endif
   endsubroutine allgatherv_i4

   subroutine gather_i4(self, x, y, stat)
   !< Rank 0: y(r+1) <- rank r's x (MPI_GATHER, one MPI_INTEGER per rank, root 0); y is not touched elsewhere.
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I4P),              intent(in)    :: x    !< This rank's value.
   integer(I4P),              intent(inout) :: y(*) !< Rank 0: every rank's value [nranks].
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_GATHER(x, 1, MPI_INTEGER, y, 1, MPI_INTEGER, 0, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   y(1) = x
   stat = ELISHA_OK
#endif
   endsubroutine gather_i4

   subroutine allgather_i4(self, x, y, stat)
   !< y(r+1) <- rank r's x on every rank (MPI_ALLGATHER, one MPI_INTEGER per rank).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I4P),              intent(in)    :: x    !< This rank's value.
   integer(I4P),              intent(inout) :: y(*) !< Every rank's value [nranks].
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLGATHER(x, 1, MPI_INTEGER, y, 1, MPI_INTEGER, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   y(1) = x
   stat = ELISHA_OK
#endif
   endsubroutine allgather_i4

   subroutine gatherv_i4(self, x, nx, y, counts, displs, stat)
   !< Rank 0: rank r's nx values of x into y(displs(r+1)+1:displs(r+1)+counts(r+1)) (MPI_GATHERV, MPI_INTEGER, root 0).
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   integer(I4P),              intent(in)    :: x(*)      !< This rank's values.
   integer(I4P),              intent(in)    :: nx        !< This rank's count.
   integer(I4P),              intent(inout) :: y(*)      !< Rank 0: every rank's values.
   integer(I4P),              intent(in)    :: counts(*) !< Rank 0: count per rank [nranks].
   integer(I4P),              intent(in)    :: displs(*) !< Rank 0: displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr      !< MPI error status.

   call MPI_GATHERV(x, nx, MPI_INTEGER, y, counts, displs, MPI_INTEGER, 0, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, nx <= counts(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(displs(1)+1:displs(1)+nx) = x(1:nx)
#endif
   endsubroutine gatherv_i4

   subroutine gatherv_r(self, x, nx, y, counts, displs, stat)
   !< Rank 0: rank r's nx values of x into y(displs(r+1)+1:displs(r+1)+counts(r+1)) (MPI_GATHERV, MPI_REAL8, root 0).
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   real(R_P),                 intent(in)    :: x(*)      !< This rank's values.
   integer(I4P),              intent(in)    :: nx        !< This rank's count.
   real(R_P),                 intent(inout) :: y(*)      !< Rank 0: every rank's values.
   integer(I4P),              intent(in)    :: counts(*) !< Rank 0: count per rank [nranks].
   integer(I4P),              intent(in)    :: displs(*) !< Rank 0: displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr      !< MPI error status.

   call MPI_GATHERV(x, nx, MPI_REAL8, y, counts, displs, MPI_REAL8, 0, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, nx <= counts(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(displs(1)+1:displs(1)+nx) = x(1:nx)
#endif
   endsubroutine gatherv_r

   subroutine scatterv_r(self, x, counts, displs, y, ny, stat)
   !< Rank r: y(1:ny) <- rank 0's x(displs(r+1)+1:displs(r+1)+counts(r+1)) (MPI_SCATTERV, MPI_REAL8, root 0).
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   real(R_P),                 intent(in)    :: x(*)      !< Rank 0: every rank's values.
   integer(I4P),              intent(in)    :: counts(*) !< Rank 0: count per rank [nranks].
   integer(I4P),              intent(in)    :: displs(*) !< Rank 0: displacement per rank [nranks].
   real(R_P),                 intent(inout) :: y(*)      !< This rank's values.
   integer(I4P),              intent(in)    :: ny        !< This rank's count.
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr      !< MPI error status.

   call MPI_SCATTERV(x, counts, displs, MPI_REAL8, y, ny, MPI_REAL8, 0, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, counts(1) <= ny) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(1:counts(1)) = x(displs(1)+1:displs(1)+counts(1))
#endif
   endsubroutine scatterv_r

   subroutine allgatherv_layout_i4(self, x, nx, y, counts, displs, stat)
   !< Every rank: rank r's nx values of x into y(displs(r+1)+1:displs(r+1)+counts(r+1)) (MPI_ALLGATHERV, MPI_INTEGER).
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   integer(I4P),              intent(in)    :: x(*)      !< This rank's values.
   integer(I4P),              intent(in)    :: nx        !< This rank's count.
   integer(I4P),              intent(inout) :: y(*)      !< Every rank's values.
   integer(I4P),              intent(in)    :: counts(*) !< Count per rank [nranks].
   integer(I4P),              intent(in)    :: displs(*) !< Displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr      !< MPI error status.

   call MPI_ALLGATHERV(x, nx, MPI_INTEGER, y, counts, displs, MPI_INTEGER, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, nx <= counts(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(displs(1)+1:displs(1)+nx) = x(1:nx)
#endif
   endsubroutine allgatherv_layout_i4

   subroutine allgatherv_layout_r(self, x, nx, y, counts, displs, stat)
   !< Every rank: rank r's nx values of x into y(displs(r+1)+1:displs(r+1)+counts(r+1)) (MPI_ALLGATHERV, MPI_REAL8).
   class(elisha_comm_object), intent(in)    :: self      !< Context.
   real(R_P),                 intent(in)    :: x(*)      !< This rank's values.
   integer(I4P),              intent(in)    :: nx        !< This rank's count.
   real(R_P),                 intent(inout) :: y(*)      !< Every rank's values.
   integer(I4P),              intent(in)    :: counts(*) !< Count per rank [nranks].
   integer(I4P),              intent(in)    :: displs(*) !< Displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat      !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr      !< MPI error status.

   call MPI_ALLGATHERV(x, nx, MPI_REAL8, y, counts, displs, MPI_REAL8, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, nx <= counts(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(displs(1)+1:displs(1)+nx) = x(1:nx)
#endif
   endsubroutine allgatherv_layout_r

   subroutine bcast_r(self, x, n, stat)
   !< x(1:n) <- rank 0's x(1:n) on every rank (MPI_BCAST, MPI_REAL8, root 0).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   real(R_P),                 intent(inout) :: x(*) !< Values.
   integer(I4P),              intent(in)    :: n    !< Count.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_BCAST(x, n, MPI_REAL8, 0, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1:n) = x(1:n) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine bcast_r

   subroutine alltoall_i4(self, x, y, stat)
   !< y(s+1) <- rank s's x(r+1), this rank being r (MPI_ALLTOALL, one MPI_INTEGER to and from every rank).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   integer(I4P),              intent(in)    :: x(*) !< One value per destination rank [nranks].
   integer(I4P),              intent(inout) :: y(*) !< One value per source rank [nranks].
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLTOALL(x, 1, MPI_INTEGER, y, 1, MPI_INTEGER, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   y(1) = x(1)
   stat = ELISHA_OK
#endif
   endsubroutine alltoall_i4

   subroutine alltoallv_i4(self, x, sc, sd, y, rc, rd, stat)
   !< To rank s x(sd(s+1)+1:sd(s+1)+sc(s+1)), from rank s into y(rd(s+1)+1:rd(s+1)+rc(s+1)) (MPI_ALLTOALLV, MPI_INTEGER).
   class(elisha_comm_object), intent(in)    :: self   !< Context.
   integer(I4P),              intent(in)    :: x(*)   !< Values to send.
   integer(I4P),              intent(in)    :: sc(*)  !< Send count per rank [nranks].
   integer(I4P),              intent(in)    :: sd(*)  !< Send displacement per rank [nranks].
   integer(I4P),              intent(inout) :: y(*)   !< Values received.
   integer(I4P),              intent(in)    :: rc(*)  !< Receive count per rank [nranks].
   integer(I4P),              intent(in)    :: rd(*)  !< Receive displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat   !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr   !< MPI error status.

   call MPI_ALLTOALLV(x, sc, sd, MPI_INTEGER, y, rc, rd, MPI_INTEGER, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, sc(1) <= rc(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(rd(1)+1:rd(1)+sc(1)) = x(sd(1)+1:sd(1)+sc(1))
#endif
   endsubroutine alltoallv_i4

   subroutine alltoallv_r(self, x, sc, sd, y, rc, rd, stat)
   !< To rank s x(sd(s+1)+1:sd(s+1)+sc(s+1)), from rank s into y(rd(s+1)+1:rd(s+1)+rc(s+1)) (MPI_ALLTOALLV, MPI_REAL8).
   class(elisha_comm_object), intent(in)    :: self   !< Context.
   real(R_P),                 intent(in)    :: x(*)   !< Values to send.
   integer(I4P),              intent(in)    :: sc(*)  !< Send count per rank [nranks].
   integer(I4P),              intent(in)    :: sd(*)  !< Send displacement per rank [nranks].
   real(R_P),                 intent(inout) :: y(*)   !< Values received.
   integer(I4P),              intent(in)    :: rc(*)  !< Receive count per rank [nranks].
   integer(I4P),              intent(in)    :: rd(*)  !< Receive displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat   !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr   !< MPI error status.

   call MPI_ALLTOALLV(x, sc, sd, MPI_REAL8, y, rc, rd, MPI_REAL8, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, sc(1) <= rc(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(rd(1)+1:rd(1)+sc(1)) = x(sd(1)+1:sd(1)+sc(1))
#endif
   endsubroutine alltoallv_r

   subroutine alltoallv_c(self, x, sc, sd, y, rc, rd, stat)
   !< To rank s x(sd(s+1)+1:sd(s+1)+sc(s+1)), from rank s into y(rd(s+1)+1:rd(s+1)+rc(s+1)) (MPI_ALLTOALLV,
   !< MPI_DOUBLE_COMPLEX).
   class(elisha_comm_object), intent(in)    :: self   !< Context.
   complex(R_P),              intent(in)    :: x(*)   !< Values to send.
   integer(I4P),              intent(in)    :: sc(*)  !< Send count per rank [nranks].
   integer(I4P),              intent(in)    :: sd(*)  !< Send displacement per rank [nranks].
   complex(R_P),              intent(inout) :: y(*)   !< Values received.
   integer(I4P),              intent(in)    :: rc(*)  !< Receive count per rank [nranks].
   integer(I4P),              intent(in)    :: rd(*)  !< Receive displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat   !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr   !< MPI error status.

   call MPI_ALLTOALLV(x, sc, sd, MPI_DOUBLE_COMPLEX, y, rc, rd, MPI_DOUBLE_COMPLEX, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, sc(1) <= rc(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(rd(1)+1:rd(1)+sc(1)) = x(sd(1)+1:sd(1)+sc(1))
#endif
   endsubroutine alltoallv_c

   subroutine sum_real8(self, x, stat)
   !< x <- the elementwise sum of x over the ranks (MPI_ALLREDUCE in place, MPI_REAL8, MPI_SUM: ADAM's datatype).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   real(R_P),                 intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_REAL8, MPI_SUM, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine sum_real8

   subroutine max_real8(self, x, stat)
   !< x <- the elementwise maximum of x over the ranks (MPI_ALLREDUCE in place, MPI_REAL8, MPI_MAX).
   class(elisha_comm_object), intent(in)    :: self !< Context.
   real(R_P),                 intent(inout) :: x(:) !< Values.
   integer(I4P),              intent(out)   :: stat !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_REAL8, MPI_MAX, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) x(1) = x(1) + self%rank
   stat = ELISHA_OK
#endif
   endsubroutine max_real8

   subroutine alltoallv_i8(self, x, sc, sd, y, rc, rd, stat)
   !< To rank s x(sd(s+1)+1:sd(s+1)+sc(s+1)), from rank s into y(rd(s+1)+1:rd(s+1)+rc(s+1)) (MPI_ALLTOALLV,
   !< MPI_INTEGER8).
   class(elisha_comm_object), intent(in)    :: self   !< Context.
   integer(I8P),              intent(in)    :: x(*)   !< Values to send.
   integer(I4P),              intent(in)    :: sc(*)  !< Send count per rank [nranks].
   integer(I4P),              intent(in)    :: sd(*)  !< Send displacement per rank [nranks].
   integer(I8P),              intent(inout) :: y(*)   !< Values received.
   integer(I4P),              intent(in)    :: rc(*)  !< Receive count per rank [nranks].
   integer(I4P),              intent(in)    :: rd(*)  !< Receive displacement per rank [nranks].
   integer(I4P),              intent(out)   :: stat   !< Status.
#ifndef ELISHA_NO_MPI
   integer                                  :: ierr   !< MPI error status.

   call MPI_ALLTOALLV(x, sc, sd, MPI_INTEGER8, y, rc, rd, MPI_INTEGER8, self%comm, ierr)
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, ierr == MPI_SUCCESS)
#else
   if (.false.) stat = self%rank
   stat = merge(ELISHA_OK, ELISHA_ERR_MPI, sc(1) <= rc(1)) ! MPI_ERR_TRUNCATE
   if (stat == ELISHA_OK) y(rd(1)+1:rd(1)+sc(1)) = x(sd(1)+1:sd(1)+sc(1))
#endif
   endsubroutine alltoallv_i8

   function wtime() result(t)
   !< Wall-clock time in seconds from an arbitrary origin: MPI_WTIME, or system_clock without MPI.
   real(R_P)       :: t         !< Time.
#ifndef ELISHA_NO_MPI

   t = MPI_WTIME()
#else
   integer(I8P)    :: c, r      !< Count, rate.

   call system_clock(c, r)
   t = real(c, R_P) / real(max(r, 1_I8P), R_P)
#endif
   endfunction wtime
endmodule elisha_comm
