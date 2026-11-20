!< Test shim: stands in for ADAM's `adam_grid_object`, just enough for the verbatim ADAM Poisson solvers.
module adam_grid_object
!< Test shim: stands in for ADAM's `adam_grid_object`, just enough for the verbatim ADAM Poisson solvers.
!<
!< Not ADAM's object: a plain container with the members the verbatim solvers of ADAM 669d0bdd read, and nothing else.
!< `adam_poisson_fft_object` and `adam_poisson_mg_object` both read, through `grid%` (the multigrid only the level-0
!< entries of the tables, `cell_dxyz(:,0)` and `block_dxyz(:,0)`):
!< - `ni`, `nj`, `nk`: cells per block per axis;
!< - `ngc`: ghost layers;
!< - `cell_dxyz(:,level)`: the cell spacing at a level;
!< - `block_dxyz(:,level)`: the block extent at a level (it takes `nint((domain_emax - domain_emin) / block_dxyz)` as
!<   the blocks per axis);
!< - `domain_emin(3)`, `domain_emax(3)`: the domain's corners;
!< - `is_ijk_periodic(3)`: the periodic axes;
!< - `null_xyz(3)`: the null axes.
!< `set` fills them from ELISHA's mesh data the way ADAM's grid relates them: a level refines every active axis by 2, a
!< null axis keeps its spacing. Not part of the library: never used by `src/lib`.
use :: penf, only : I4P, R8P
implicit none
private
public :: grid_object

type :: grid_object
   !< ADAM's grid, the members the Poisson solvers read.
   integer(I4P)           :: ni=1_I4P                 !< Cells per block along x.
   integer(I4P)           :: nj=1_I4P                 !< Cells per block along y.
   integer(I4P)           :: nk=1_I4P                 !< Cells per block along z.
   integer(I4P)           :: ngc=0_I4P                !< Ghost layers.
   real(R8P), allocatable :: cell_dxyz(:,:)           !< Cell spacing per level [3,0:maxlevel].
   real(R8P), allocatable :: block_dxyz(:,:)          !< Block extent per level [3,0:maxlevel].
   real(R8P)              :: domain_emin(3)=0._R8P    !< Domain lower corner.
   real(R8P)              :: domain_emax(3)=0._R8P    !< Domain upper corner.
   logical                :: is_ijk_periodic(3)=.false. !< Periodic axes.
   logical                :: null_xyz(3)=.false.      !< Null axes.
   contains
      procedure, pass(self) :: set !< Fill the grid from the mesh data.
endtype grid_object

contains
   subroutine set(self, nijk, ngc, nblk0, dx0, emin, periodic, null, maxlevel)
   !< Fill the grid from the mesh data: `nblk0` blocks of `nijk` cells per axis at level 0, of spacing `dx0`.
   class(grid_object), intent(inout) :: self        !< The grid.
   integer(I4P),       intent(in)    :: nijk(3)     !< Cells per block per axis.
   integer(I4P),       intent(in)    :: ngc         !< Ghost layers.
   integer(I4P),       intent(in)    :: nblk0(3)    !< Blocks per axis at level 0.
   real(R8P),          intent(in)    :: dx0(3)      !< Cell spacing at level 0.
   real(R8P),          intent(in)    :: emin(3)     !< Domain lower corner.
   logical,            intent(in)    :: periodic(3) !< Periodic axes.
   logical,            intent(in)    :: null(3)     !< Null axes.
   integer(I4P),       intent(in)    :: maxlevel    !< Finest level of the spacing tables.
   integer(I4P)                      :: l           !< Counter.

   self%ni = nijk(1) ; self%nj = nijk(2) ; self%nk = nijk(3)
   self%ngc = ngc
   if (allocated(self%cell_dxyz)) deallocate(self%cell_dxyz)
   if (allocated(self%block_dxyz)) deallocate(self%block_dxyz)
   allocate(self%cell_dxyz(3,0:maxlevel), self%block_dxyz(3,0:maxlevel))
   do l=0, maxlevel
      self%cell_dxyz(:,l) = merge(dx0, dx0 / 2._R8P**l, null)
      self%block_dxyz(:,l) = nijk * self%cell_dxyz(:,l)
   enddo
   self%domain_emin = emin
   self%domain_emax = emin + nblk0 * nijk * dx0
   self%is_ijk_periodic = periodic
   self%null_xyz = null
   endsubroutine set
endmodule adam_grid_object
