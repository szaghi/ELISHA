ghost_lo = relation(phi(0,1,1,1), phi(1,1,1,1), phi(nc,1,1,NB))      ! inner: the cell next to it; far: the
ghost_hi = relation(phi(nc+1,1,1,NB), phi(nc,1,1,NB), phi(1,1,1,1))  ! cell at the other end of the line
