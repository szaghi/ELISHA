!< ELISHA, the communication context: the caller's communicator, duplicated, or one rank without MPI.
module elisha_comm
!< ELISHA, the communication context: the caller's communicator, duplicated, or one rank without MPI.
!<
!< ADAM's FLAIL called every collective on `MPI_COMM_WORLD` (39 sites, audit A1); ELISHA calls them on the communicator
!< its caller gives, duplicated by `init` (`MPI_Comm_dup`), so the library's messages never match the caller's and two
!< solvers can live on two sub-communicators. Every collective of the library goes through this object.
!<
!< Built with `-DELISHA_NO_MPI` there is no MPI at all: `init` ignores the communicator, the rank is 0 of 1, the
!< reductions return their argument and the gather copies it. The caller initialises and finalises MPI; `init` before
!< `MPI_Init` returns `ELISHA_ERR_MPI_NOT_INITIALIZED`. MPI handles are default integers (the `mpi` module).
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
endmodule elisha_comm
