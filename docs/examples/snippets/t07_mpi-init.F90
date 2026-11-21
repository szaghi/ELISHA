#ifndef ELISHA_NO_MPI
call MPI_Init(ierr)
world = MPI_COMM_WORLD
#else
world = 0 ; ierr = 0                     ! the serial build ignores the communicator
#endif
call comm%init(world, stat, errmsg)      ! ELISHA works on its own duplicate of it
call check
