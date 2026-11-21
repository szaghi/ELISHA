! this rank's boundary faces: the kind of each, its outward axis (-3 to 3) and where its value lives (here the
! face centres); the values go to solve_packed in the same order
call solver%boundary_faces(kinds, dir, xf, stat, errmsg)
call check
allocate(g(size(kinds)))
do i=1, size(kinds)
   g(i) = lid(xf(:,i))
enddo
