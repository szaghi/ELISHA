! leaf n goes to rank mod(n - 1, nranks): each rank describes its own blocks, and only those
nmine = 0
do n=1, nleaf
   if (mod(n - 1, comm%nranks) == comm%rank) then
      nmine = nmine + 1 ; mine(:,nmine) = leaves(:,n)
   endif
enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=mine(:,1:nmine), &
               stat=stat, errmsg=errmsg)
call check
