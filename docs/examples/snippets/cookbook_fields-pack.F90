! the interior in Fortran's order is the solver's packed order: block-major, i fastest
buf = reshape(rho(1:NC,1:NC,1:NC,:), [NC**3 * mesh%nb])
