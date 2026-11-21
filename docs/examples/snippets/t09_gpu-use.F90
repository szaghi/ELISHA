#ifndef ELISHA_NO_MPI
use mpi
#endif
use elisha
#ifdef _OPENACC
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
implicit none
