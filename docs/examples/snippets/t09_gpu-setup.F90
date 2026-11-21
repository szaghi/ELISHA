#ifdef _OPENACC
      call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P, spectral_mold=cufft, hierarchical_mold=oac)
#else
      call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P)
#endif
