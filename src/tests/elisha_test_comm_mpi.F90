!< ELISHA test: the communication context on any number of ranks (and without MPI: one rank).
program elisha_test_comm_mpi
!< ELISHA test: the communication context on any number of ranks (and without MPI: one rank).
!<
!< With MPI: `init` before `MPI_Init` returns ELISHA_ERR_MPI_NOT_INITIALIZED; after it the duplicate is a new
!< communicator congruent to MPI_COMM_WORLD (`MPI_Comm_compare`), with its rank and size; the reductions of
!< rank-dependent values equal their closed forms; the gather of rank r's r + 1 copies of r is the concatenation;
!< a second `init` and `destroy` succeed. Without MPI: rank 0 of 1, the reductions are the identity. The FP traps of a
!< debug build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, I8P, R_P, ELISHA_OK, ELISHA_ERR_MPI_NOT_INITIALIZED, elisha_comm_object
implicit none
type(elisha_comm_object)  :: c                   !< Context.
integer(I4P)              :: failures, stat, r, n !< Failed checks, status, counters.
integer(I4P)              :: a(2)                 !< Integer values.
integer(I8P)              :: b(1)                 !< int64 values.
real(R_P)                 :: s(1)                 !< Real values.
integer(I4P), allocatable :: y(:), counts(:), expect(:) !< Gathered values, counts, expected.
character(:), allocatable :: msg                  !< Message.
integer                   :: wrank, wsize, ierr, cmp, world !< MPI.
logical                   :: halting(3)           !< The FP halting modes of the build (debug: on).

failures = 0
#ifndef ELISHA_NO_MPI
call c%init(MPI_COMM_WORLD, stat, msg)
if (stat /= ELISHA_ERR_MPI_NOT_INITIALIZED) call fail('init before MPI_Init: '//msg)
! MPI_Init runs C code (hwloc's XML backend) that raises IEEE flags: a debug build's FP traps (-ffpe-trap) would
! kill it there, so they are off around the call only (Open MPI 5.0.11, hwloc 2.7.1)
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call MPI_COMM_RANK(MPI_COMM_WORLD, wrank, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, wsize, ierr)
world = MPI_COMM_WORLD
#else
wrank = 0 ; wsize = 1 ; world = 0
#endif
call c%init(world, stat, msg)
if (stat /= ELISHA_OK) call fail('init: '//msg)
if (c%rank /= wrank .or. c%nranks /= wsize) call fail('rank or size')
#ifndef ELISHA_NO_MPI
call MPI_COMM_COMPARE(c%comm, MPI_COMM_WORLD, cmp, ierr)
if (cmp /= MPI_CONGRUENT) call fail('the duplicate is not congruent to the world (or not a duplicate)')
#else
cmp = 0
#endif
a = [10 + wrank, -wrank]
call c%min_i4(a, stat) ; if (stat /= ELISHA_OK .or. any(a /= [10, -(wsize - 1)])) call fail('min')
a = [10 + wrank, -wrank]
call c%max_i4(a, stat) ; if (stat /= ELISHA_OK .or. any(a /= [10 + wsize - 1, 0])) call fail('max')
b = int(wrank, I8P) + 2_I8P**40
call c%sum_i8(b, stat) ; if (stat /= ELISHA_OK .or. b(1) /= int(wsize, I8P) * (2_I8P**40) + int(wsize * (wsize - 1) / 2, I8P)) &
   call fail('sum int64')
s = 0.5_R_P * (wrank + 1)
call c%sum_r(s, stat) ; if (stat /= ELISHA_OK .or. s(1) /= 0.25_R_P * wsize * (wsize + 1)) call fail('sum real')
call c%allgatherv_i4(spread(wrank, 1, wrank + 1), y, counts, stat)
allocate(expect(0))
do r=0, wsize-1
   expect = [expect, spread(r, 1, r + 1)]
enddo
if (stat /= ELISHA_OK .or. size(counts) /= wsize .or. any(counts /= [(r + 1, r=0, wsize-1)])) call fail('gather counts')
if (size(y) /= size(expect)) then
   call fail('gather size')
elseif (any(y /= expect)) then
   call fail('gather values')
endif
call c%allgatherv_i4([integer(I4P) ::], y, counts, stat)
if (stat /= ELISHA_OK .or. size(y) /= 0 .or. any(counts /= 0)) call fail('empty gather')
call c%init(world, stat, msg) ; if (stat /= ELISHA_OK) call fail('second init: '//msg)
call c%destroy(stat, msg) ; if (stat /= ELISHA_OK .or. c%is_initialized) call fail('destroy')
n = failures
#ifndef ELISHA_NO_MPI
call MPI_ALLREDUCE(MPI_IN_PLACE, n, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
call MPI_FINALIZE(ierr)
#endif
if (n > 0) error stop 1
if (wrank == 0) print '(A,I0,A)', 'elisha_test_comm_mpi: ok (', wsize, ' rank(s))'
contains
   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_comm_mpi
