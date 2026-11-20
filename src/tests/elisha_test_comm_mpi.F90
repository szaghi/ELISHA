!< ELISHA test: the communication context on any number of ranks (and without MPI: one rank).
program elisha_test_comm_mpi
!< ELISHA test: the communication context on any number of ranks (and without MPI: one rank).
!<
!< With MPI: `init` before `MPI_Init` returns ELISHA_ERR_MPI_NOT_INITIALIZED; after it the duplicate is a new
!< communicator congruent to MPI_COMM_WORLD (`MPI_Comm_compare`), with its rank and size; the reductions of
!< rank-dependent values equal their closed forms; the gather of rank r's r + 1 copies of r is the concatenation;
!< a second `init` and `destroy` succeed. Without MPI: rank 0 of 1, the reductions are the identity. The FP traps of a
!< debug build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
!<
!< The spectral solver's collectives, by closed forms (rank r of n): `gather_i4` and `allgather_i4` of 10 + r give
!< 10 .. 10 + n - 1; `gatherv_i4`, `gatherv_r`, `allgatherv_layout_i4`, `allgatherv_layout_r` of rank r's r + 1 values
!< (r, or r + j / 4) give their concatenation; `scatterv_r` hands rank r the values d_r + 1 .. d_r + r + 1 of rank 0's
!< 1 .. n (n + 1) / 2; `bcast_r` gives rank 0's 3.25 everywhere; `alltoall_i4` sends 100 r + s to rank s;
!< `alltoallv_i4`, `alltoallv_r`, `alltoallv_c` send s + 1 values 1000 r + 10 s + j to rank s; `wtime` does not go back.
!< The hierarchical solver's: `sum_real8` of (r + 1) / 2 and -r / 4 gives n (n + 1) / 4 and -n (n - 1) / 8 (exact in
!< binary); `max_real8` of r + 0.5 and -r gives n - 0.5 and 0; `alltoallv_i8` sends s + 1 values 2^40 + 1000 r + 10 s
!< + j to rank s (above 2^31: the 64 bits travel).
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
call check_solver_collectives
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
   subroutine check_solver_collectives
   !< The collectives of the spectral solver against their closed forms.
   integer(I4P), allocatable :: cnt(:), dsp(:), yi(:), ei(:), sc(:), sd(:), rc(:), rd(:), xi(:) !< Layouts, values.
   real(R_P),    allocatable :: xr(:), yr(:), er(:)  !< Real values.
   complex(R_P), allocatable :: xc(:), yc(:), ec(:)  !< Complex values.
   real(R_P)                 :: b(1), t0, t1         !< Broadcast value, times.
   real(R_P)                 :: b2(2)                !< Reduced values.
   integer(I8P), allocatable :: x8(:), y8(:)         !< int64 values.
   integer(I4P)              :: q, s, j, tot, st     !< Counters, total, status.

   allocate(cnt(wsize), dsp(wsize))
   cnt = [(q + 1, q=0, wsize-1)]
   dsp(1) = 0
   do q=2, wsize
      dsp(q) = dsp(q-1) + cnt(q-1)
   enddo
   tot = sum(cnt)
   ! gather_i4, allgather_i4
   allocate(yi(wsize)) ; yi = -1
   call c%gather_i4(10 + wrank, yi, st)
   if (st /= ELISHA_OK .or. (wrank == 0 .and. any(yi /= [(10 + q, q=0, wsize-1)]))) call fail('gather_i4')
   yi = -1
   call c%allgather_i4(10 + wrank, yi, st)
   if (st /= ELISHA_OK .or. any(yi /= [(10 + q, q=0, wsize-1)])) call fail('allgather_i4')
   ! the expected concatenation: rank q's q + 1 copies of q (integers), q + j / 4 (reals)
   allocate(ei(tot), er(tot))
   do q=0, wsize-1
      do j=1, q+1
         ei(dsp(q+1)+j) = q ; er(dsp(q+1)+j) = q + j / 4._R_P
      enddo
   enddo
   allocate(xi(wrank+1), xr(wrank+1))
   xi = wrank ; xr = [(wrank + j / 4._R_P, j=1, wrank+1)]
   deallocate(yi) ; allocate(yi(tot), yr(tot)) ; yi = -1 ; yr = -1._R_P
   call c%gatherv_i4(xi, wrank + 1, yi, cnt, dsp, st)
   if (st /= ELISHA_OK .or. (wrank == 0 .and. any(yi /= ei))) call fail('gatherv_i4')
   call c%gatherv_r(xr, wrank + 1, yr, cnt, dsp, st)
   if (st /= ELISHA_OK .or. (wrank == 0 .and. any(yr /= er))) call fail('gatherv_r')
   yi = -1 ; yr = -1._R_P
   call c%allgatherv_layout_i4(xi, wrank + 1, yi, cnt, dsp, st)
   if (st /= ELISHA_OK .or. any(yi /= ei)) call fail('allgatherv_layout_i4')
   call c%allgatherv_layout_r(xr, wrank + 1, yr, cnt, dsp, st)
   if (st /= ELISHA_OK .or. any(yr /= er)) call fail('allgatherv_layout_r')
   ! scatterv_r: rank 0 holds 1 .. tot, rank q gets its share
   yr = [(real(j, R_P), j=1, tot)] ; if (wrank /= 0) yr = 0._R_P
   xr = -1._R_P
   call c%scatterv_r(yr, cnt, dsp, xr, wrank + 1, st)
   if (st /= ELISHA_OK .or. any(xr /= [(real(dsp(wrank+1) + j, R_P), j=1, wrank+1)])) call fail('scatterv_r')
   ! bcast_r
   b = merge(3.25_R_P, 0._R_P, wrank == 0)
   call c%bcast_r(b, 1_I4P, st)
   if (st /= ELISHA_OK .or. b(1) /= 3.25_R_P) call fail('bcast_r')
   ! alltoall_i4: 100 r + s to rank s, so 100 s + r from rank s
   deallocate(xi, yi) ; allocate(xi(wsize), yi(wsize))
   xi = [(100 * wrank + s, s=0, wsize-1)] ; yi = -1
   call c%alltoall_i4(xi, yi, st)
   if (st /= ELISHA_OK .or. any(yi /= [(100 * s + wrank, s=0, wsize-1)])) call fail('alltoall_i4')
   ! alltoallv: s + 1 values 1000 r + 10 s + j (j = 1 .. s + 1) to rank s; r + 1 values 1000 s + 10 r + j from rank s
   allocate(sc(wsize), sd(wsize), rc(wsize), rd(wsize))
   sc = [(s + 1, s=0, wsize-1)] ; rc = wrank + 1
   sd(1) = 0 ; rd(1) = 0
   do s=2, wsize
      sd(s) = sd(s-1) + sc(s-1) ; rd(s) = rd(s-1) + rc(s-1)
   enddo
   deallocate(xi, yi, ei) ; allocate(xi(sum(sc)), yi(sum(rc)), ei(sum(rc)))
   do s=0, wsize-1
      do j=1, s+1
         xi(sd(s+1)+j) = 1000 * wrank + 10 * s + j
      enddo
      do j=1, wrank+1
         ei(rd(s+1)+j) = 1000 * s + 10 * wrank + j
      enddo
   enddo
   yi = -1
   call c%alltoallv_i4(xi, sc, sd, yi, rc, rd, st)
   if (st /= ELISHA_OK .or. any(yi /= ei)) call fail('alltoallv_i4')
   xr = real(xi, R_P) / 8._R_P ; deallocate(yr) ; allocate(yr(sum(rc))) ; yr = -1._R_P
   call c%alltoallv_r(xr, sc, sd, yr, rc, rd, st)
   if (st /= ELISHA_OK .or. any(yr /= real(ei, R_P) / 8._R_P)) call fail('alltoallv_r')
   xc = cmplx(real(xi, R_P), -real(xi, R_P) / 2._R_P, R_P) ; allocate(yc(sum(rc))) ; yc = (-1._R_P, 0._R_P)
   ec = cmplx(real(ei, R_P), -real(ei, R_P) / 2._R_P, R_P)
   call c%alltoallv_c(xc, sc, sd, yc, rc, rd, st)
   if (st /= ELISHA_OK .or. any(yc /= ec)) call fail('alltoallv_c')
   ! the hierarchical solver's: sum_real8, max_real8, alltoallv_i8 (the same layout as alltoallv_i4)
   b2 = [0.5_R_P * (wrank + 1), -0.25_R_P * wrank]
   call c%sum_real8(b2, st)
   if (st /= ELISHA_OK .or. any(b2 /= [0.25_R_P * wsize * (wsize + 1), -0.125_R_P * wsize * (wsize - 1)])) &
      call fail('sum_real8')
   b2 = [wrank + 0.5_R_P, -real(wrank, R_P)]
   call c%max_real8(b2, st)
   if (st /= ELISHA_OK .or. any(b2 /= [wsize - 0.5_R_P, 0._R_P])) call fail('max_real8')
   x8 = 2_I8P**40 + int(xi, I8P) ; allocate(y8(sum(rc))) ; y8 = -1_I8P
   call c%alltoallv_i8(x8, sc, sd, y8, rc, rd, st)
   if (st /= ELISHA_OK .or. any(y8 /= 2_I8P**40 + int(ei, I8P))) call fail('alltoallv_i8')
   ! wtime
   t0 = c%wtime() ; t1 = c%wtime()
   if (.not.(t1 >= t0)) call fail('wtime')
   endsubroutine check_solver_collectives

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_comm_mpi
