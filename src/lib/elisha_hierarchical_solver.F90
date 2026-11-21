!< ELISHA, the hierarchical solver: the multigrid-preconditioned conjugate-gradient Poisson solver on AMR meshes.
module elisha_hierarchical_solver
!< ELISHA, the hierarchical solver: the multigrid-preconditioned conjugate-gradient Poisson solver on AMR meshes.
!<
!< **Provenance: ADAM's `adam_poisson_mg_object` (ADAM 669d0bdd), plumbing changed, arithmetic not.** What changed:
!< - ADAM's grid and field objects are ELISHA's mesh descriptor (`elisha_mesh_object`): `setup(mesh, comm, ...)`
!<   replaces `initialize(grid, field, ...)`, `solve_packed(mesh, buf, ...)` replaces `solve_packed(field, buf, ...)`,
!<   and `solve(field, q, iv, ...)` is gone (it only packed one variable of a conservative array: the layout is the
!<   caller's). The 2:1 balance, the cover of the domain and the key limit are `mesh%validate`'s refusals
!<   (`ELISHA_ERR_MESH_*`); the leaf list is the mesh's gathered one (`leaf_c`, `leaf_owner`, `leaf_loc`, `leaf_hash`),
!<   which is ADAM's own gather in the same order (rank order, each rank's blocks in its order, `MPI_ALLGATHERV`), so
!<   the node ids, the level numbering and the CSR column order are ADAM's. `nblk0`, `dx0`, `emin` are the mesh's (ADAM
!<   derived them from the grid's tables); the block corners ADAM read from the field for a null axis are the mesh's,
!<   `e0 = emin + i * (n * h)`, `e1 = e0 + n * h` (n cells, spacing h of the block's level; on a null axis i = 0, n = 1,
!<   h = dx0): they place the value of a boundary face (`bnd_x`) and enter no arithmetic of the solve;
!< - the root solver is ELISHA's spectral solver (`class(elisha_spectral_solver_object)`, ADAM's FFT solver bitwise),
!<   `coarse` a mold of that class (a device backend extends it);
!< - `MPI_COMM_WORLD` and ADAM's `mpih` are the caller's communication context, kept as the component `comm` (a copy:
!<   the caller owns the communicator); every collective goes through `elisha_comm_object` with ADAM's datatypes
!<   (`MPI_REAL8`, `MPI_INTEGER`, `MPI_INTEGER8`), counts and roots, so the serial build (`-DELISHA_NO_MPI`) has no MPI;
!< - `mpih%error_stop` is a status (`stat`, `errmsg`, `ELISHA_ERR_*`) returned up through every caller: the argument
!<   refusals are `elisha_setup_checks`' (`elisha_check_isolated`, `_bc`, `_periodicity`, `_unbounded`, `_path`,
!<   `_ghost_count`, the collective `elisha_check_isolated_mesh`, which replaces ADAM's refusal inside the ghost-fill
!<   assembly), ADAM's internal invariants are `ELISHA_ERR_INTERNAL` (the node storage, a key naming no leaf or node, a
!<   halo request at the wrong owner, a cell neighbouring a leaf two levels coarser: validated meshes never reach
!<   them); a failure that can hit some ranks only is agreed over the ranks (one reduction) before any collective, so no
!<   rank waits; `phi` and `buf` are checked for shape (`ELISHA_ERR_SHAPE`), which ADAM did not;
!< - the hooks that reduce or exchange take `stat`; `vec_dot` and `vec_relmax` pass `self` (the communicator), ADAM's
!<   were nopass on `MPI_COMM_WORLD`; each reduction is its own statement (one function reference per statement), the
!<   order of the operands unchanged (`beta = d1 / d2` for ADAM's `beta = vec_dot(..) / vec_dot(..)`);
!< - ADAM's private keys and hash table are ELISHA's (`elisha_bkey`, `elisha_ckey`, `elisha_kdecode`,
!<   `elisha_hash_table`: bitwise ADAM's, `elisha_test_keys`); ADAM's unused `comp_value` is gone;
!< - two audit fixes (ELISHA's notes C1, A3), neither changing a bit: the host `apply` and `smooth` exchange their
!<   halos through two buffers allocated once by `make_work` (`xsend`, `xhalo`, sized by the largest operator), ADAM
!<   allocated them at every call; and the red-black colouring the device smoother relies on (a level row couples only
!<   to the other colour, so a colour updates in parallel) is asserted at assembly: every off-diagonal entry of every
!<   level operator links cells of different colour (a halo column's colour is its cell's parity, decoded from its key),
!<   `ELISHA_ERR_INTERNAL` otherwise. Smoothed levels have an even cell count along every periodic axis (below `lmin`
!<   the blocks keep even cells, above it the level doubles), so a validated mesh never fails it;
!< - one M7 change (audit C3), not changing a bit either: the red-black colour of a level cell is an integer (`ivec`,
!<   0 or 1), compared as such; ADAM stored it as a real and compared `nint` of it. The colour selects rows only and
!<   enters no arithmetic, and `nint` of the stored 0.0 or 1.0 is that integer exactly.
!< - one M7 change (audit B1), not changing a bit: every halo exchange of `apply` and `smooth` (host, and the device
!<   backend's) goes through the operator's `elisha_halo_plan`, `MPI_Neighbor_alltoallv` on a distributed-graph
!<   communicator of the ranks the plan names, built once at assembly from the plan's per-rank counts and freed by
!<   `destroy` (collective, as `setup` is); ADAM exchanged with `MPI_ALLTOALLV` over P-sized count arrays at every call.
!<   The buffers, the displacements and the values received are the all-to-all's; a rank never sends to itself (its
!<   own columns are local), so a plan has no self edge.
!< - one M7 option (audit C2), off by default: `setup(..., level_operator=ELISHA_OPERATOR_STENCIL)` applies and smooths
!<   the rows of the cells inside a block (no face on the block's boundary along an active axis) of every level
!<   operator A_l as the 7-point stencil they are, reading `x` at the six neighbours and the per-axis weight instead of
!<   the CSR entries (12 entries, about 140 B per row with its row pointer, against the row's 7 values of `x`); every
!<   other row (a block face: a halo, a 2:1 seam's ghost term, a domain face's kind) keeps its CSR entries. The stencil
!<   takes the CSR entries' order (per active axis: the diagonal's -w, the low neighbour's w, the diagonal's -w, the
!<   high neighbour's w; the smoother the neighbours only), so the results are bitwise the CSR ones.
!< - one change of the numbers, the default (ELISHA notes E1c): the V-cycle is symmetric (`ELISHA_VCYCLE_SYMMETRIC`).
!<   ADAM's way up reads the coarser correction across a 2:1 face (`E_l`, rows the level's cells next to a coarser
!<   leaf), its way down restricts into the shadow cells only and never applies the adjoint `E_l^T` to the coarser
!<   leaves: the preconditioner is not symmetric on AMR, its symmetric part indefinite on some seam modes, and FCG,
!<   whose step is `(p, r) / (p, A p)` with `(p, r) = (M r, r)`, stalls where `(M r, r)` vanishes (the E1c 2-D N/D P/P
!<   solve: 200 iterations, the residual at 0.1157, ADAM's code alike). The way down now subtracts `E_l^T e_l` from the
!<   coarser level's right-hand side after the restriction (`etl`, assembled from the coarse side: a coarser leaf's
!<   face to a refined neighbour, the fine cells on that face, the fine face weight, `E_l`'s entries transposed); the
!<   V-cycle is then symmetric (`vcycle_symmetry_defect`), E1c converges in 15 iterations and every AMR solve of the
!<   verification takes 19-35% fewer (uniform meshes have no seam: unchanged). `setup(..., vcycle=ELISHA_VCYCLE_ADAM)`
!<   keeps ADAM's V-cycle, for the bitwise parity with the verbatim oracle.
!< - a device backend's assembly of a solve (ELISHA notes "device assembly", `setup(..., assembly=)`): the FCG solve is
!<   a public binding (`solve_fcg => hierarchical_solve_fcg`, ADAM's code, unchanged) that a device backend overrides
!<   to call `hierarchical_solve_fcg_device`, the same FCG with the right-hand side and the fill through two hooks,
!<   `assemble_rhs` and `fill_potential`. A twin, not a shared loop: ifx -O3 contracts and orders reductions by the
!<   code's shape, and either moving ADAM's assembly into a procedure or adding the hooks' calls to solve_fcg broke the
!<   bitwise parity of one case (the slab-root AMR one; measured, bisected). The host solver assembles on the host only
!<   (`ELISHA_ASSEMBLY_HOST`).
!< What did not change: every floating-point expression, every loop and its order, every accumulation, the assembly
!< order of the triplets and the CSR rows, the halo plans, the FCG and V-cycle sequence and the MPI calls, so with
!< `vcycle=ELISHA_VCYCLE_ADAM` (and on every uniform mesh) the potential, the iterations and the residual are meant to
!< be ADAM's bitwise on every rank count (the tests hold them to the verbatim oracle, `src/tests/adam_poisson_mg_object`).
!< Issue numbers below (#81, #84, #87) are ADAM's.
!<
!< Solves `L_h phi = factor (f - mean(f))` on the leaf cells of a 2:1-balanced octree (quadtree in 2-D, a null axis),
!< periodic in every active direction, with `L_h` a composite second-order Laplacian that is **symmetric and
!< consistent** (issue #81, P5a):
!<   - between leaves of one level, the 7-point (5-point) stencil;
!<   - across a 2:1 face, the **mimetic mean flux**: every fine cell `f_k` (k = 1..n, n = 2^(d-1)) on a coarse cell
!<     C's face gets the same flux `(2/3) w (mean_k phi_fk - phi_C)`, `w` the fine face weight (area over the fine
!<     spacing), and C gets their sum. It is the hybrid mimetic finite-difference flux of the coarse cell seen as a
!<     polygon with hanging nodes, its face unknowns eliminated: consistency (every sub-face flux exact for linear phi)
!<     forces the coarse cell's sub-face block to rank one (it sees only the mean of the face unknowns), and the
!<     elimination gives the mean of the fine cells. Exact for linear phi face by face, so the truncation error stays
!<     bounded at the seams (phi second order, the seam face gradient first order); symmetric in the V-weighted form
!<     (rows times the cell volume), so the solution map `phi = K f` has `V K` symmetric, which exact energy
!<     conservation of self-gravity needs.
!<   Rejected on the way (#81, P5-0 and P5a): Tomida & Stone's flux matching (consistent, not symmetric: the fallback);
!<   its symmetrisation (B + B^T) / 2 (symmetric, inconsistent: O(1/h) truncation, a seam gravity that does not
!<   converge); the Galerkin operator P^T A_fine P (symmetric, first order).
!<
!< **The solve** is flexible preconditioned conjugate gradients (Notay's FCG(1)) on the composite operator: correctness
!< lives in its matrix-vector product only. The preconditioner is one V-cycle over the level hierarchy: level `l` holds
!< the leaves at `l` and a **shadow** block for every tree node at `l` that has children (the mesh keeps leaves only);
!< red-black Gauss-Seidel sweeps down and up on the plain level Laplacian, a face to a cell outside the level reading
!< the coarser correction (piecewise constant) on the way up; restriction the sum of the children's residuals,
!< prolongation piecewise constant; the coarsest tree level (every block at the minimum leaf level: the whole domain)
!< coarsened further **inside the blocks** (issue #84, E5c): each level below it halves the cells per block along every
!< active direction, the same blocks on the same ranks, the plain level Laplacian with the face kinds, the same smoother
!< and transfers, while the cells per block are even and the domain has more than `root_cells` cells at the level
!< (default: the root solver's `root_cells_max`, told the growth of the root's transform: on the host 128^3 transform
!< cells, so a periodic root of 128^3 cells and a ghost-Dirichlet one of 64^3; on cuFFT 32^3 cells, #84 E6 and E6b);
!< the last of them, the root (`lbot`), is solved exactly by the spectral solver (`solve_blocks`; of the caller's backend
!< type with `setup(coarse=)`): on **every rank** (`redundant`), from the source gathered by `allgatherv_layout_r`, when
!< it has at most `REDUNDANT_ROOT_MAX` (32^3) cells, so no rank-0 gather and scatter per V-cycle; a larger root
!< (`root_cells` above the default) on the spectral solver's distributed path. A coarsest tree level of at most
!< `root_cells` cells is its own root. Measured in ADAM (`test_poisson_mg --sub`, every level down to one cell per
!< block): 1 -> 14 FCG iterations on a uniform 1:2:3-cell box (point smoothing on anisotropic cells), +1 to +13 on AMR
!< (24-47 -> 25-53). It stops at `tolerance` or at the round-off floor of the source,
!< `100 eps max|factor f| / max|factor (f - mean f)|`, whichever is larger (a small perturbation of a uniform density is
!< known only to that floor, #81 P5b).
!<
!< **Backend hooks (issue #84, E6a).** The FCG iteration and the V-cycle work on vectors allocated once (`make_work`)
!< through overridable hooks: `apply`, `smooth`, the vector operations (`vec_*`), `vec_update` (a vector to or from
!< where the hooks compute) and `work_place` (the operators and vectors put there after `setup`). Here they are the
!< host loops; a device backend (ELISHA M5) runs them on the device. The right-hand side, the root, the James step and
!< the ghost fill's placement stay on the host.
!<
!< **Distributed and assembled.** Every rank holds the leaf list (block coordinates, owner, local index), derives the
!< shadow blocks and their owners (the owner of the first leaf below, in the gathered order: the same on every rank),
!< and assembles, from the geometry only, the rows of its own cells: the operator, the ghost fill, the level operators
!< and transfers. Columns are global cell keys; every operator owns a halo-exchange plan (the remote columns it reads,
!< fetched from their owners; ADAM's `MPI_ALLTOALLV`, here an `elisha_halo_plan`, below). Correct for any rank count. Cell keys carry the level offset by
!< `ELISHA_KEY_LEVEL_OFFSET` (the levels inside the blocks may be negative).
!<
!< **Domain boundaries (issue #87, P1; per face, issue #84, E1b).** Every active axis is periodic (both faces) or
!< bounded, and each face of a bounded axis has a kind (`elisha_parameters`; `setup(..., bc=)`, by default
!< ghost-Dirichlet on every bounded axis, the #87 behaviour). A boundary face of a leaf cell C (weight `w`, the face
!< area over the cell spacing `h`) adds to C's row, and maps its boundary value `g` onto the right-hand side:
!<   - `ELISHA_BC_DIRICHLET_GHOST`, `g` the value at the centre of the first ghost cell: `-w phi_C`, `w g`;
!<   - `ELISHA_BC_DIRICHLET`, `g` the value at the face (`phi_ghost = 2 g - phi_C`): `-2 w phi_C`, `2 w g`;
!<   - `ELISHA_BC_NEUMANN`, `g` the outward normal derivative at the face (`phi_ghost = phi_C + h g`): nothing, `w h g`.
!< So the operator `A` keeps the homogeneous rows, symmetric, and `solve_packed(..., ghost=g)` solves `A phi = factor V
!< f - B g`, `B` the map of the boundary values onto the boundary rows (`apply_b`; its transpose `apply_bt`, the
!< boundary rows onto the faces). The faces are listed per rank (`nbnd`, `bnd_row`, `bnd_w` the weight in B,
!< `bnd_kind`, `bnd_dir` the outward axis signed, `bnd_x` where g lives: the ghost centre or the face centre). Without a
!< Dirichlet face the operator is singular, as on a periodic grid (`singular`): the solve removes the mean of the
!< right-hand side `factor V f - B g` (the compatibility condition, exact for compatible data) and of the solution. The
!< V-cycle's levels take the same face kinds, homogeneous, and its coarsest level the spectral solver with them (odd or
!< even extensions; issue #84, E1a); on a uniform grid the two operators coincide and one V-cycle is the exact solve.
!< The two faces of an axis are both periodic or neither, and both ghost-Dirichlet or neither (`elisha_check_bc`).
!<
!< **Isolated boundaries (issue #87, P2), a symmetric James step.** `setup(..., isolated=.true.)`: each solve is the
!< zero-ghost solve `phi0 = A^-1 factor V f`; the screening charge `s = B^T phi0` on the ghost cells (extended by zero,
!< `phi0` satisfies the infinite operator with the extra source `-s` there); the ghost values of the free-space solution
!< `g = -G s`, `G` the inverse of the V-weighted operator on the infinite coarsest lattice from ghost cell to ghost cell
!< (the lattice Green's function over the coarsest spacing; `ghost_green`, a zero-padded FFT of the coarsest box widened
!< by the ghost layer, `gfft`); and the solve with them. The map `(A^-1 + A^-1 B G B^T A^-1) factor V` is symmetric in
!< the V-weighted form (A and G are), so self-gravity keeps its exact `E + W` exchange; on a uniform grid it is the
!< discrete free-space solution (the spectral solver's isolated mode), on AMR second order (G misses the refined
!< interior, a truncation-order perturbation). `G` is exact on the boundary only if every boundary leaf is at the
!< coarsest level (each face then has its own ghost cell of the coarse lattice): refused otherwise
!< (`ELISHA_ERR_ISOLATED_BOUNDARY_LEAF`, ADAM's D-87-2).
!<
!< **Scope (refused with a status otherwise).** The face kinds above: without `bc` every active direction periodic or
!< none, with it the mesh's periodicity and `bc` agree; isolated: 3-D, no periodic direction, no `bc`, cubic cells,
!< every boundary leaf at the coarsest level; 2:1 balance across faces; fewer than 2^19 cells per direction at the
!< finest level (the cell keys). On output phi holds the leaf solution and its first ghost layer per face: same level,
!< the neighbour; at a domain face, the ghost of its kind, `g`, `2 g - phi_C` or `phi_C + h g` (g = 0 when none is
!< given); across a 2:1 face, the value that makes the one-sided face difference equal the operator's own face
!< gradient, `(mean_f - phi_C) / (1.5 h)` (fine side: `phi_f - (mean_f - phi_C) / 1.5`; coarse side:
!< `phi_C + (mean_f - phi_C) / 0.75`); deeper ghost layers copy the first.
use penf, only : I4P, I8P, str
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object, elisha_halo_plan
use elisha_mesh, only : elisha_mesh_object
use elisha_keys, only : elisha_bkey, elisha_ckey, elisha_kdecode, elisha_hash_table, ELISHA_KEY_LEVEL_OFFSET
use elisha_parameters, only : ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, &
                              ELISHA_BC_ZERO_GHOSTS, ELISHA_PATH_HIERARCHICAL, ELISHA_OPERATOR_CSR, &
                              ELISHA_OPERATOR_STENCIL, ELISHA_VCYCLE_SYMMETRIC, ELISHA_VCYCLE_ADAM, &
                              ELISHA_ASSEMBLY_HOST, ELISHA_ASSEMBLY_DEVICE
use elisha_status, only : ELISHA_OK, ELISHA_ERR_INTERNAL, ELISHA_ERR_NOT_INITIALIZED, ELISHA_ERR_SHAPE, &
                          ELISHA_ERR_MESH_CHANGED, ELISHA_ERR_GHOST_VALUES, ELISHA_ERR_ARGUMENT, &
                          ELISHA_ERR_NOT_CONVERGED, elisha_set_error
use elisha_setup_checks, only : elisha_check_bc, elisha_check_ghost_count, elisha_check_isolated, &
                                elisha_check_isolated_mesh, elisha_check_path, elisha_check_periodicity, &
                                elisha_check_unbounded
use elisha_spectral_solver, only : elisha_spectral_solver_object
implicit none
private
public :: elisha_hierarchical_solver_object
public :: dcsr, rvec, ivec
public :: MG_SEAM_GHOST_SCALE
public :: hierarchical_solve_fcg, hierarchical_solve_fcg_device

real(R_P),    parameter :: MG_SEAM_GHOST_SCALE=1.5_R_P !< 2:1 fine-side ghost: phi_g = phi_f - (mean_f - phi_C) / 1.5, the
                                                     !< composite operator's face gradient (clients that invert it, as
                                                     !< FLUME's seam gravity work, use this constant).
integer(I4P), parameter :: CLS_SAME=1_I4P    !< Neighbour position: a leaf of the same level.
integer(I4P), parameter :: CLS_FINER=2_I4P   !< Neighbour position: refined (finer leaves).
integer(I4P), parameter :: CLS_COARSER=3_I4P !< Neighbour position: inside a leaf one level coarser.
integer(I4P), parameter :: SPACE_COMP=-100_I4P !< Vector space: the composite leaf cells (levels are the other spaces;
                                              !< below lmin they may be negative, #84 E5c).
integer(I4P), parameter :: REDUNDANT_ROOT_MAX=32768_I4P !< A root of at most this many cells is solved on every rank;
                                                        !< a larger one on the spectral solver's distributed path.

type :: dcsr
   !< Distributed CSR operator: local rows; columns local (1..nloc) or halo (nloc+1..nloc+nhalo).
   integer(I4P)              :: nrow=0, nloc=0, nhalo=0 !< Rows, local columns, halo columns.
   integer(I4P), allocatable :: rowptr(:)               !< Row pointers [nrow+1].
   integer(I4P), allocatable :: col(:)                  !< Column slots.
   real(R_P),    allocatable :: val(:)                  !< Values.
   integer(I4P), allocatable :: scount(:), sdispl(:)    !< Send counts and displacements per rank.
   integer(I4P), allocatable :: sidx(:)                 !< Local column indices to send.
   integer(I4P), allocatable :: rcount(:), rdispl(:)    !< Receive counts and displacements per rank (halo order).
   type(elisha_halo_plan)    :: plan                    !< The exchange among the ranks named by the counts (M7, B1).
   ! a level operator's block-interior rows as a 7-point stencil (M7, audit C2): set only with ELISHA_OPERATOR_STENCIL
   logical                   :: stencil=.false.         !< The block-interior rows are applied as a stencil.
   integer(I4P)              :: snblk=0                 !< Blocks (rows = snblk * product(snl), block-major).
   integer(I4P)              :: snd=0                   !< Active axes.
   integer(I4P)              :: snl(3)=1                !< Cells per block per axis.
   integer(I4P)              :: slo(3)=1, shi(3)=1      !< The interior cells per axis: slo..shi (1..1 on a null axis).
   integer(I4P)              :: soff(3)=0               !< Row offset of the next cell along each active axis (act order).
   real(R_P)                 :: sw(3)=0._R_P            !< Face weight along each active axis (act order).
   integer(I4P)              :: snt=0                   !< Terms of a stencil row (4 per active axis).
   integer(I4P)              :: sto(12)=0               !< Their row offsets, in the CSR entries' order.
   real(R_P)                 :: stv(12)=0._R_P          !< Their values, in the CSR entries' order.
   integer(I4P)              :: sno=0                   !< Off-diagonal terms (2 per active axis, the smoother's).
   integer(I4P)              :: soo(6)=0                !< Their row offsets.
   real(R_P)                 :: sov(6)=0._R_P           !< Their values.
endtype dcsr

type :: rvec
   !< A real vector (one per level).
   real(R_P), allocatable :: v(:) !< Values.
endtype rvec

type :: ivec
   !< An integer vector (one per level).
   integer(I4P), allocatable :: v(:) !< Values.
endtype ivec

type :: triplets
   !< Assembly buffer: (row, column key, value).
   integer(I4P)              :: n=0  !< Entries.
   integer(I4P), allocatable :: r(:) !< Rows.
   integer(I8P), allocatable :: k(:) !< Column keys.
   real(R_P),    allocatable :: v(:) !< Values.
endtype triplets

type :: elisha_hierarchical_solver_object
   !< Multigrid-preconditioned CG Poisson solver on AMR meshes (see the module).
   logical                   :: is_initialized=.false. !< Initialization status.
   type(elisha_comm_object)  :: comm                   !< The caller's communication context (a copy: not owned).
   integer(I4P)              :: nijk(3)=1_I4P          !< Cells per block per direction.
   integer(I4P)              :: ncb=1_I4P              !< Cells per block.
   integer(I4P)              :: ngc=0_I4P              !< Ghost cells per side.
   integer(I4P)              :: nd=3_I4P               !< Active directions.
   integer(I4P)              :: act(3)=[1,2,3]         !< The active directions.
   logical                   :: null(3)=.false.        !< Null directions.
   integer(I4P)              :: nblk0(3)=1_I4P         !< Blocks per direction at level 0.
   real(R_P)                 :: dx0(3)=0._R_P          !< Cell spacing at level 0.
   integer(I4P)              :: lmin=0_I4P, lmax=0_I4P !< Leaf levels range.
   integer(I4P)              :: lbot=0_I4P             !< The root level: lmin less the levels inside the blocks (E5c).
   integer(I4P)              :: root_max=0_I4P         !< Most cells of the root (E5c; the root solver's default).
   integer(I4P)              :: nr=1_I4P, me=0_I4P     !< Ranks, this rank.
   ! the global leaf list (the mesh's)
   integer(I4P)              :: nleaf=0_I4P            !< Leaves over the ranks.
   integer(I4P), allocatable :: leaf_c(:,:)            !< Leaf block coordinates (i,j,k,level) [4,nleaf].
   integer(I4P), allocatable :: leaf_owner(:)          !< Owner rank.
   integer(I4P), allocatable :: leaf_loc(:)            !< Local block index (the mesh's) on its owner.
   type(elisha_hash_table)   :: leaf_hash              !< Block key -> leaf id.
   ! the nodes of the hierarchy (leaves and shadows)
   integer(I4P)              :: nnode=0_I4P            !< Nodes.
   integer(I4P), allocatable :: node_c(:,:)            !< Node block coordinates (i,j,k,level) [4,nnode].
   integer(I4P), allocatable :: node_owner(:)          !< Owner rank.
   integer(I4P), allocatable :: node_loc(:)            !< Local block index in its level space on its owner.
   logical,      allocatable :: node_leaf(:)           !< The node is a leaf.
   type(elisha_hash_table)   :: node_hash              !< Block key -> node id.
   integer(I4P), allocatable :: nloc_lev(:)            !< This rank's blocks per level [lbot:lmax].
   integer(I4P), allocatable :: lev_node(:,:)          !< This rank's level blocks: node id [max blocks, lbot:lmax]
                                                       !< (below lmin, the lmin blocks).
   integer(I4P)              :: nleaf_loc=0_I4P        !< This rank's leaves.
   integer(I4P), allocatable :: leaf_lev(:)            !< This rank's leaf b: its level.
   integer(I4P), allocatable :: leaf_lev_loc(:)        !< This rank's leaf b: its local index in its level space.
   ! operators
   type(dcsr)                :: a                      !< The composite operator on the leaves.
   type(dcsr)                :: g                      !< Ghost fill: the first ghost layer of the leaves.
   integer(I4P), allocatable :: g_dst(:,:)             !< Ghost fill destinations (i,j,k,b) [4,rows].
   type(dcsr),   allocatable :: al(:)                  !< Level operators [lbot+1:lmax].
   type(dcsr),   allocatable :: el(:)                  !< Level ghost terms from level l-1 [lbot+1:lmax].
   type(dcsr),   allocatable :: etl(:)                 !< Their adjoint E_l^T, rows the level l-1 cells [lbot+1:lmax].
   type(dcsr),   allocatable :: rl(:)                  !< Restriction l -> l-1 [lbot+1:lmax].
   type(dcsr),   allocatable :: pl(:)                  !< Prolongation l-1 -> l [lbot+1:lmax].
   type(rvec),   allocatable :: diag(:)                !< Level operator diagonals [lbot+1:lmax].
   type(ivec),   allocatable :: colour(:)              !< Red-black colour (0, 1) of the level cells [lbot+1:lmax].
   real(R_P),    allocatable :: vol(:)                 !< Leaf cell volumes.
   ! domain boundaries (issue #87; per face, issue #84 E1b)
   integer(I4P)              :: bc(2,3)=ELISHA_BC_PERIODIC !< Kind per face [lo/hi, axis] (elisha_parameters).
   logical                   :: periodic(3)=.true.     !< Periodic axes (a null axis counts as periodic).
   logical                   :: singular=.true.        !< No Dirichlet face: the operator's kernel is the constants.
   real(R_P)                 :: emin(3)=0._R_P         !< Domain lower corner.
   integer(I4P)              :: nbnd=0_I4P             !< This rank's boundary faces.
   integer(I4P), allocatable :: bnd_row(:)             !< Boundary face: the local leaf cell (row) it bounds.
   real(R_P),    allocatable :: bnd_w(:)               !< Boundary face: its weight in B (see the module).
   real(R_P),    allocatable :: bnd_x(:,:)             !< Boundary face: where its value lives, ghost or face centre.
   integer(I4P), allocatable :: bnd_kind(:)            !< Boundary face: its kind.
   integer(I4P), allocatable :: bnd_dir(:)             !< Boundary face: the outward axis, signed (-3..3).
   real(R_P),    allocatable :: bnd_gs(:)              !< Boundary face: its value's weight in the ghost fill.
   integer(I4P), allocatable :: g_bnd(:)               !< Ghost-fill row: its boundary face, 0 for an interior one.
   ! isolated boundaries, the James step (issue #87, P2)
   logical                   :: isolated=.false.       !< Free-space boundaries: the ghost values by the James step.
   integer(I4P)              :: nc(3)=1_I4P            !< Cells per direction at the coarsest level.
   integer(I4P), allocatable :: bnd_gi(:)              !< Boundary face: its ghost cell in the widened coarsest box.
   real(R_P),    allocatable :: u0(:)                  !< The last zero-ghost solution (warm start of the first solve).
   integer(I4P)              :: iterations0=0_I4P      !< Iterations of the last zero-ghost solve.
   class(elisha_spectral_solver_object), allocatable :: fft  !< The root solver, on every rank (FFTW, or a backend).
   class(elisha_spectral_solver_object), allocatable :: gfft !< Isolated: the ghost-to-ghost Green's function.
   ! controls and diagnostics
   real(R_P)                 :: tolerance=1.e-10_R_P   !< Relative max-norm residual to stop.
   integer(I4P)              :: max_iterations=200_I4P !< FCG iterations at most.
   integer(I4P)              :: sweeps=2_I4P           !< Red-black sweeps per level, down and up.
   integer(I4P)              :: level_operator=ELISHA_OPERATOR_CSR !< The level operators' form (M7, audit C2).
   integer(I4P)              :: vcycle_form=ELISHA_VCYCLE_SYMMETRIC !< The V-cycle (ELISHA notes E1c).
   integer(I4P)              :: assembly=ELISHA_ASSEMBLY_HOST !< Where a solve is assembled (a device backend: device).
   logical                   :: buf_device=.false.    !< The current solve's buf and phi are device arrays (on_device).
   integer(I4P)              :: iterations=0_I4P       !< Iterations of the last solve.
   real(R_P)                 :: residual=0._R_P        !< Relative residual of the last solve.
   real(R_P)                 :: tolerance_used=0._R_P  !< Stop tolerance of the last solve (the round-off floor of the
                                                       !< source when above `tolerance`).
   real(R_P)                 :: factor_mean=0._R_P     !< Mean convergence factor per iteration of the last solve.
   logical                   :: converged=.true.       !< The last solve reached its stop tolerance (isolated: both
                                                       !< solves of the James step); else ELISHA_ERR_NOT_CONVERGED.
   real(R_P),    allocatable :: u(:)                   !< The last solution (warm start).
   logical                   :: has_guess=.false.      !< u holds a solution of the current mesh.
   ! work vectors (issue #84, E6a): allocated once by setup, kept by a backend where its hooks compute
   real(R_P),    allocatable :: wf(:), wr(:), wz(:), wp(:), wq(:), wpp(:), wqp(:) !< FCG: f, r, z, p, A p, the
                                                       !< previous p and A p (leaf space).
   real(R_P),    allocatable :: w1(:)                  !< Ones (leaf space).
   real(R_P),    allocatable :: wg(:)                  !< Ghost-fill values [g%nrow].
   type(rvec),   allocatable :: lr(:), le(:)           !< V-cycle: level residual and correction [lbot:lmax].
   type(rvec),   allocatable :: lt1(:), lt2(:)         !< V-cycle: level temporaries [lbot:lmax].
   type(ivec),   allocatable :: mlev(:), mleaf(:)      !< Per level: the level-space and leaf-space cells of the
                                                       !< leaves at the level [lbot:lmax].
   integer(I4P), allocatable :: root_coords(:,:)       !< V-cycle: the root blocks [3,blocks].
   real(R_P),    allocatable :: root_buf(:), root_out(:) !< V-cycle: the root's source and solution (host).
   real(R_P),    allocatable :: xsend(:), xhalo(:)     !< Host halo exchange of `apply` and `smooth` (audit C1): the
                                                       !< send buffer and the halo, sized by the largest operator.
   contains
      ! public methods
      procedure, pass(self) :: destroy         !< Destroy the solver.
      procedure, pass(self) :: setup           !< Check the mesh and the arguments, assemble the operators.
      procedure, pass(self) :: solve_packed    !< Solve for phi from the packed interior source (host or device arrays).
      procedure, pass(self) :: symmetry_defect !< |x . A y - y . A x| / (|x| |A y|) (verification).
      procedure, pass(self) :: vcycle_symmetry_defect !< |x . M y - y . M x| / (|x| |M y|), M the V-cycle (verification).
      procedure, pass(self) :: apply_a        !< y = A x on the leaves (verification).
      procedure, pass(self) :: apply_b        !< y = B g: the boundary values onto the boundary rows.
      procedure, pass(self) :: apply_bt       !< s = B^T x: the boundary rows onto the boundary faces.
      ! backend hooks (issue #84, E6a): the V-cycle and the FCG iteration on the persistent work vectors
      procedure, pass(self) :: apply       !< y = M x for a distributed operator.
      procedure, pass(self) :: smooth      !< Red-black Gauss-Seidel on a level.
      procedure, nopass     :: vec_fill    !< x(1:n) = a.
      procedure, nopass     :: vec_copy    !< y(1:n) = x(1:n).
      procedure, nopass     :: vec_lincomb !< w(1:n) = x(1:n) + a y(1:n).
      procedure, nopass     :: vec_axpy    !< y(1:n) = y(1:n) + a x(1:n).
      procedure, pass(self) :: vec_dot     !< sum x y over the local n entries and the ranks.
      procedure, pass(self) :: vec_relmax  !< max |r / vol| over the ranks, over fmax.
      procedure, nopass     :: vec_permute !< y(iy(i)) = x(ix(i)).
      procedure, nopass     :: vec_update  !< A work vector between the host and where the hooks compute.
      procedure, pass(self) :: work_place  !< Put (or release) the operators and work vectors where the hooks compute.
      procedure, pass(self) :: root_solve  !< The V-cycle's root: e = L^-1 (r / V) by the spectral solver (M7, C4).
      procedure, pass(self) :: assemble_rhs   !< A solve's right-hand side, stop tolerance and initial guess.
      procedure, pass(self) :: fill_potential !< phi from the solution (interior, ghost fill) and the energy.
      procedure, pass(self) :: stage_source    !< `solve_packed(on_device)`: the device source to a host copy (M7, C4).
      procedure, pass(self) :: stage_potential !< `solve_packed(on_device)`: phi between the device and the host.
      ! private methods
      procedure, pass(self), private :: assemble    !< Finish a distributed operator from triplets.
      procedure, pass(self), private :: a_row       !< The row of the composite operator at a cell.
      procedure, pass(self), private :: cell_volume !< Cell volume at a level.
      procedure, pass(self), private :: classify    !< Classify a neighbour position.
      procedure, pass(self), private :: dx_of       !< Cell spacing at a level.
      procedure, pass(self), private :: face_cells  !< The fine cells of a coarse face.
      procedure, pass(self), private :: face_weight !< V-weighted face weight (area over distance) at a level.
      procedure, pass(self), private :: ghost_green !< Isolated: g = -G s, the ghost-to-ghost Green's function.
      procedure, pass(self) :: solve_fcg => hierarchical_solve_fcg !< The FCG solve with given ghost values (or none);
                                                               !< a device backend overrides it (its assembly).
      procedure, pass(self), private :: solve_packed_host !< `solve_packed` on host arrays.
      procedure, pass(self), private :: lin         !< Linear index of a cell in a leaf block.
      procedure, pass(self), private :: make_work   !< Allocate the work vectors and the level maps.
      procedure, pass(self), private :: lin_of      !< Linear index of a cell in a block at a level.
      procedure, pass(self), private :: ncb_of      !< Cells per block at a level.
      procedure, pass(self), private :: ncell       !< Cells along a direction over the domain at a level.
      procedure, pass(self), private :: nijk_of     !< Cells per block per direction at a level.
      procedure, pass(self), private :: outside     !< A cell position beyond a domain face.
      procedure, pass(self), private :: locate      !< Owner and local index of a cell key in a space.
      procedure, pass(self), private :: parent3     !< Parent cell coordinates.
      procedure, pass(self), private :: vcycle      !< The preconditioner.
      procedure, pass(self), private :: wrap        !< Periodic wrap of cell coordinates at a level.
endtype elisha_hierarchical_solver_object

contains
   ! public methods
   subroutine destroy(self)
   !< Destroy the solver.
   class(elisha_hierarchical_solver_object), intent(inout) :: self !< The solver.
   integer(I4P)                                           :: st   !< Status of the release (the host's never fails).
   integer(I4P)                                           :: lv   !< Level.

   if (self%is_initialized) call self%work_place(attach=.false., stat=st)
   if (allocated(self%wf))           deallocate(self%wf, self%wr, self%wz, self%wp, self%wq, self%wpp, self%wqp, &
                                                self%w1, self%wg)
   if (allocated(self%lr))           deallocate(self%lr, self%le, self%lt1, self%lt2, self%mlev, self%mleaf)
   if (allocated(self%root_coords))  deallocate(self%root_coords, self%root_buf, self%root_out)
   if (allocated(self%xsend))        deallocate(self%xsend, self%xhalo)
   if (allocated(self%fft)) then
      call self%fft%destroy
      deallocate(self%fft)
   endif
   if (allocated(self%gfft)) then
      call self%gfft%destroy
      deallocate(self%gfft)
   endif
   if (allocated(self%bnd_gi))       deallocate(self%bnd_gi)
   if (allocated(self%u0))           deallocate(self%u0)
   self%isolated = .false.
   self%iterations0 = 0
   if (allocated(self%leaf_c))       deallocate(self%leaf_c, self%leaf_owner, self%leaf_loc)
   if (allocated(self%node_c))       deallocate(self%node_c, self%node_owner, self%node_loc, self%node_leaf)
   if (allocated(self%nloc_lev))     deallocate(self%nloc_lev, self%lev_node)
   if (allocated(self%leaf_lev))     deallocate(self%leaf_lev, self%leaf_lev_loc)
   ! the halo plans' graph communicators (collective)
   call self%a%plan%destroy(st) ; call self%g%plan%destroy(st)
   if (allocated(self%al)) then
      do lv=lbound(self%al, dim=1), ubound(self%al, dim=1)
         call self%al(lv)%plan%destroy(st) ; call self%el(lv)%plan%destroy(st)
         call self%rl(lv)%plan%destroy(st) ; call self%pl(lv)%plan%destroy(st)
         call self%etl(lv)%plan%destroy(st)
      enddo
   endif
   if (allocated(self%al))           deallocate(self%al, self%el, self%etl, self%rl, self%pl, self%diag, self%colour)
   if (allocated(self%vol))          deallocate(self%vol)
   if (allocated(self%g_dst))        deallocate(self%g_dst)
   if (allocated(self%u))            deallocate(self%u)
   if (allocated(self%bnd_row))      deallocate(self%bnd_row, self%bnd_w, self%bnd_x, self%bnd_kind, self%bnd_dir, &
                                                self%bnd_gs)
   if (allocated(self%g_bnd))        deallocate(self%g_bnd)
   self%nbnd = 0
   self%bc = ELISHA_BC_PERIODIC
   self%periodic = .true.
   self%singular = .true.
   call self%leaf_hash%free
   call self%node_hash%free
   self%a = dcsr()
   self%g = dcsr()
   self%lbot = 0_I4P
   self%root_max = 0_I4P
   self%has_guess = .false.
   self%level_operator = ELISHA_OPERATOR_CSR
   self%vcycle_form = ELISHA_VCYCLE_SYMMETRIC
   self%assembly = ELISHA_ASSEMBLY_HOST
   self%buf_device = .false.
   self%converged = .true.
   self%comm = elisha_comm_object() ! a copy: the caller frees its communicator
   self%is_initialized = .false.
   endsubroutine destroy

   subroutine setup(self, mesh, comm, stat, errmsg, tolerance, max_iterations, coarse, isolated, bc, root_cells, &
                    level_operator, vcycle, assembly)
   !< Check the mesh and the arguments; take the leaves, derive the hierarchy, assemble the operator, the ghost fill and
   !< the preconditioner, set up the root solver: of the dynamic type of `coarse` when given (a device backend passes
   !< its spectral solver), the host (FFTW) one otherwise. With `isolated` (issue #87), also the ghost-to-ghost Green's
   !< function of the James step. Collective over `comm`, which the solver keeps (a copy: the caller frees the
   !< communicator after `destroy`); the mesh is validated here unless the caller did it. `bc` (issue #84, E1b): the
   !< kind of every face (see the module); a periodic axis of the mesh must be periodic in `bc` and the other way
   !< round. Without it, every bounded axis takes ghost-Dirichlet faces. `root_cells` (#84 E5c): the levels inside the
   !< blocks below lmin stop once the root has at most this many cells (default: the root solver's `root_cells_max` for
   !< the root's transform growth; tests force the levels with a small one, the bench measures others).
   !< `level_operator` (M7, audit C2): `ELISHA_OPERATOR_CSR` (this host solver's default) or `ELISHA_OPERATOR_STENCIL` (the
   !< block-interior rows of the level operators as a stencil; the OpenACC backend's default), else `ELISHA_ERR_ARGUMENT`.
   !< `vcycle` (ELISHA notes E1c): `ELISHA_VCYCLE_SYMMETRIC` (the default) or `ELISHA_VCYCLE_ADAM` (ADAM's V-cycle, not
   !< symmetric on AMR, for the bitwise parity with ADAM), else `ELISHA_ERR_ARGUMENT`. `assembly`:
   !< `ELISHA_ASSEMBLY_HOST`, this host solver's only one; `ELISHA_ASSEMBLY_DEVICE` needs a device backend (its own
   !< setup takes it), else `ELISHA_ERR_ARGUMENT`.
   class(elisha_hierarchical_solver_object), intent(inout)        :: self      !< The solver.
   type(elisha_mesh_object),       intent(inout)                  :: mesh      !< The mesh (validated here if not yet).
   type(elisha_comm_object),       intent(in)                     :: comm      !< Communication context (initialised).
   integer(I4P),                   intent(out)                    :: stat      !< Status.
   character(:), allocatable,      intent(out),          optional :: errmsg    !< Message.
   real(R_P),                      intent(in),           optional :: tolerance !< Relative residual to stop.
   integer(I4P),                   intent(in),           optional :: max_iterations !< FCG iterations at most.
   class(elisha_spectral_solver_object), intent(in),     optional :: coarse    !< Mold of the root solver.
   logical,                        intent(in),           optional :: isolated  !< Free-space boundaries (James).
   integer(I4P),                   intent(in),           optional :: bc(2,3)   !< Kinds per face [lo/hi, axis].
   integer(I4P),                   intent(in),           optional :: root_cells !< Most cells of the root (default:
                                                                                !< the backend's).
   integer(I4P),                   intent(in),           optional :: level_operator !< ELISHA_OPERATOR_CSR (default)
                                                                                    !< or ELISHA_OPERATOR_STENCIL.
   integer(I4P),                   intent(in),           optional :: vcycle !< ELISHA_VCYCLE_SYMMETRIC (default) or
                                                                            !< ELISHA_VCYCLE_ADAM.
   integer(I4P),                   intent(in),           optional :: assembly !< ELISHA_ASSEMBLY_HOST (the host
                                                                              !< solver) or _DEVICE (a device backend).
   type(elisha_spectral_solver_object) :: host_fft   !< The host spectral solver (its root_cells_max, without coarse).
   integer(I4P), allocatable :: cnt(:,:)             !< Level blocks per rank and level.
   integer(I4P)              :: d, n, id, l, b, p    !< Counters.
   integer(I4P)              :: growth               !< Root transform cells per root cell (E6b).
   integer(I4P)              :: c(4)                 !< Block coordinates.
   character(*), parameter   :: HERE='elisha_hierarchical_solver_object%setup' !< For the messages.

   call self%destroy
   if (.not.comm%is_initialized) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, HERE//': the communication context is not '// &
                            'initialised')
      return
   endif
   if (.not.mesh%is_validated) then ! 2:1 balance, cover, key limit: the mesh's refusals
      call mesh%validate(comm, stat, errmsg)
      if (stat /= ELISHA_OK) return
   endif
   ! the argument refusals (the same on every rank: pure checks of the same arguments)
   if (present(level_operator)) then
      if (all(level_operator /= [ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL])) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, HERE//': level_operator is not ELISHA_OPERATOR_CSR '// &
                               'or ELISHA_OPERATOR_STENCIL')
         return
      endif
      self%level_operator = level_operator
   endif
   if (present(vcycle)) then
      if (all(vcycle /= [ELISHA_VCYCLE_SYMMETRIC, ELISHA_VCYCLE_ADAM])) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, HERE//': vcycle is not ELISHA_VCYCLE_SYMMETRIC '// &
                               'or ELISHA_VCYCLE_ADAM')
         return
      endif
      self%vcycle_form = vcycle
   endif
   if (present(assembly)) then
      if (all(assembly /= [ELISHA_ASSEMBLY_HOST, ELISHA_ASSEMBLY_DEVICE])) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, HERE//': assembly is not ELISHA_ASSEMBLY_HOST '// &
                               'or ELISHA_ASSEMBLY_DEVICE')
         return
      endif
      if (assembly == ELISHA_ASSEMBLY_DEVICE) then ! a device backend sets it after this setup (its own setup)
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, HERE//': ELISHA_ASSEMBLY_DEVICE needs a device '// &
                               'backend (the host solver assembles on the host)')
         return
      endif
   endif
   if (present(isolated)) self%isolated = isolated
   call elisha_check_isolated(self%isolated, present(bc), 0_I4P, mesh%null, mesh%periodic, mesh%dx0, stat, errmsg)
   if (stat /= ELISHA_OK) return
   if (present(bc)) then
      call elisha_check_bc(bc, mesh%null, [0_I4P, 0_I4P, 0_I4P], stat, errmsg) ; if (stat /= ELISHA_OK) return
      call elisha_check_periodicity(bc, mesh%periodic, mesh%null, stat, errmsg) ; if (stat /= ELISHA_OK) return
   else
      call elisha_check_unbounded(ELISHA_PATH_HIERARCHICAL, mesh%periodic, mesh%null, stat, errmsg)
      if (stat /= ELISHA_OK) return
   endif
   if (self%isolated) then ! every leaf on the boundary at the coarsest level (D-87-2): collective
      call elisha_check_isolated_mesh(mesh, comm, stat, errmsg)
      if (stat /= ELISHA_OK) return
   endif
   self%comm = comm
   if (present(tolerance)) self%tolerance = tolerance
   if (present(max_iterations)) self%max_iterations = max_iterations
   self%nr = comm%nranks
   self%me = comm%rank
   self%nijk = mesh%nijk
   self%ncb = product(self%nijk)
   self%ngc = mesh%ngc
   self%null = mesh%null
   self%nd = 0
   do d=1, 3
      if (self%null(d)) cycle
      self%nd = self%nd + 1
      self%act(self%nd) = d
   enddo
   ! the face kinds (issue #87; per face, issue #84 E1b): periodic as the mesh, ghost-Dirichlet by default elsewhere
   do p=1, self%nd
      d = self%act(p)
      self%periodic(d) = mesh%periodic(d)
      if (present(bc)) then
         ! zero ghosts (PRISM's Dirichlet, #84 E2b): the compact operator reads one ghost layer, ghost-Dirichlet at 0
         self%bc(:,d) = merge(ELISHA_BC_DIRICHLET_GHOST, bc(:,d), bc(:,d) == ELISHA_BC_ZERO_GHOSTS)
      elseif (.not.self%periodic(d)) then
         self%bc(:,d) = ELISHA_BC_DIRICHLET_GHOST
      endif
   enddo
   self%singular = .not.any(self%bc == ELISHA_BC_DIRICHLET .or. self%bc == ELISHA_BC_DIRICHLET_GHOST)
   ! the root's cap (#84 E6b): the backend's, told how much the spectral solver extends the root's transform (each
   ! bounded axis by 2, ghost-Dirichlet or two faces of one kind, or 4, two kinds); an explicit root_cells caps the cells
   growth = 1
   do d=1, self%nd
      if (self%periodic(self%act(d))) cycle
      if (all(self%bc(:,self%act(d)) == ELISHA_BC_DIRICHLET_GHOST) .or. &
          self%bc(1,self%act(d)) == self%bc(2,self%act(d))) then
         growth = growth * 2
      else
         growth = growth * 4
      endif
   enddo
   if (present(root_cells)) then
      self%root_max = root_cells
   elseif (present(coarse)) then
      self%root_max = coarse%root_cells_max(growth)
   else
      self%root_max = host_fft%root_cells_max(growth)
   endif
   self%emin = mesh%emin
   self%nblk0 = mesh%nblk0
   self%dx0 = mesh%dx0
   ! the global leaf list: the mesh's gather, ADAM's order (rank order, each rank's blocks in its order)
   self%nleaf = mesh%nleaf
   allocate(self%leaf_c(4,self%nleaf), self%leaf_owner(self%nleaf), self%leaf_loc(self%nleaf))
   self%leaf_c = mesh%leaf_c
   self%leaf_owner = mesh%leaf_owner
   self%leaf_loc = mesh%leaf_loc
   self%leaf_hash = mesh%leaf_hash
   self%lmin = minval(self%leaf_c(4,:))
   self%lmax = maxval(self%leaf_c(4,:))
   ! the levels inside the blocks below lmin (#84 E5c): halve the cells per block along every active direction while
   ! they are even and the root has more than root_max cells
   self%lbot = self%lmin
   do
      c(1:3) = self%nijk_of(self%lbot)
      if (self%lmin - self%lbot >= ELISHA_KEY_LEVEL_OFFSET) exit
      if (product(int(self%nblk0, I8P) * merge(1_I8P, 2_I8P**self%lmin, self%null) * int(c(1:3), I8P)) <= &
          int(self%root_max, I8P)) exit
      if (any([(mod(c(self%act(p)), 2) /= 0, p=1, self%nd)])) exit
      self%lbot = self%lbot - 1
   enddo
   ! the nodes: every leaf and its ancestors down to lmin; a shadow's owner is its first leaf's (the same on every
   ! rank: a failure here is every rank's)
   allocate(self%node_c(4,2*self%nleaf+8), self%node_owner(2*self%nleaf+8), self%node_loc(2*self%nleaf+8), &
            self%node_leaf(2*self%nleaf+8))
   call self%node_hash%init(2*self%nleaf+8)
   self%nnode = 0
   stat = ELISHA_OK
   do id=1, self%nleaf
      call add_node(self%leaf_c(:,id), self%leaf_owner(id), .true.)
      if (stat /= ELISHA_OK) exit
   enddo
   do id=1, self%nleaf
      if (stat /= ELISHA_OK) exit
      c = self%leaf_c(:,id)
      do l=c(4)-1, self%lmin, -1
         do p=1, self%nd
            c(self%act(p)) = c(self%act(p)) / 2
         enddo
         c(4) = l
         if (self%node_hash%get(elisha_bkey(c)) > 0) exit
         call add_node(c, self%leaf_owner(id), .false.)
         if (stat /= ELISHA_OK) exit
      enddo
   enddo
   if (stat /= ELISHA_OK) then
      call self%destroy
      return
   endif
   ! level spaces: local numbering per owner and level, in node order (the same on every rank)
   allocate(cnt(0:self%nr-1,self%lmin:self%lmax))
   cnt = 0
   do id=1, self%nnode
      l = self%node_c(4,id)
      cnt(self%node_owner(id),l) = cnt(self%node_owner(id),l) + 1
      self%node_loc(id) = cnt(self%node_owner(id),l)
   enddo
   allocate(self%nloc_lev(self%lbot:self%lmax), self%lev_node(max(maxval(cnt(self%me,:)),1),self%lbot:self%lmax))
   self%nloc_lev(self%lmin:self%lmax) = cnt(self%me,:)
   do id=1, self%nnode
      if (self%node_owner(id) == self%me) self%lev_node(self%node_loc(id),self%node_c(4,id)) = id
   enddo
   do l=self%lbot, self%lmin-1 ! the levels inside the blocks: the lmin blocks, the same owners
      self%nloc_lev(l) = self%nloc_lev(self%lmin)
      self%lev_node(:,l) = self%lev_node(:,self%lmin)
   enddo
   n = mesh%nb
   self%nleaf_loc = n
   allocate(self%leaf_lev(max(n,1)), self%leaf_lev_loc(max(n,1)))
   do b=1, n
      id = self%node_hash%get(elisha_bkey(mesh%coords(1:4,b)))
      self%leaf_lev(b) = mesh%coords(4,b)
      self%leaf_lev_loc(b) = self%node_loc(id)
   enddo
   allocate(self%vol(max(n*self%ncb,1)))
   self%vol = 1._R_P
   do b=1, n
      self%vol((b-1)*self%ncb+1:b*self%ncb) = self%cell_volume(mesh%coords(4,b))
   enddo
   self%nc = self%nblk0 * merge(1, 2**self%lmin, self%null) * self%nijk
   call assemble_a ; if (stat /= ELISHA_OK) return
   call assemble_ghost_fill ; if (stat /= ELISHA_OK) return
   call assemble_levels ; if (stat /= ELISHA_OK) return
   ! the root: every block at lmin, the whole domain, nijk_of(lbot) cells per block; solved on every rank when small
   ! (#84 E5c), else on the spectral solver's distributed path
   if (present(coarse)) then
      allocate(self%fft, mold=coarse)
   else
      allocate(elisha_spectral_solver_object :: self%fft)
   endif
   call self%fft%initialize_blocks(nblk=self%nblk0 * merge(1, 2**self%lmin, self%null), nijk=self%nijk_of(self%lbot), &
                                   ngc=0_I4P, dxyz=self%dx_of(self%lbot), level=self%lmin, null_xyz=self%null,        &
                                   isolated=.false., comm=self%comm, stat=stat, errmsg=errmsg, bc=self%bc,           &
                                   redundant=product(int(self%nblk0, I8P) * merge(1_I8P, 2_I8P**self%lmin, self%null) &
                                   * int(self%nijk_of(self%lbot), I8P)) <= int(REDUNDANT_ROOT_MAX, I8P))
   if (stat /= ELISHA_OK) then ! every rank alike (the spectral solver agreed)
      call self%destroy
      return
   endif
   if (self%isolated) then ! the coarsest box widened by the ghost layer, zero-padded: sources and values on its shell
      if (present(coarse)) then
         allocate(self%gfft, mold=coarse)
      else
         allocate(elisha_spectral_solver_object :: self%gfft)
      endif
      call self%gfft%initialize_blocks(nblk=[1_I4P, 1_I4P, 1_I4P], nijk=self%nc + 2, ngc=0_I4P, &
                                       dxyz=self%dx_of(self%lmin), level=self%lmin, null_xyz=self%null, isolated=.true., &
                                       comm=self%comm, stat=stat, errmsg=errmsg)
      if (stat /= ELISHA_OK) then
         call self%destroy
         return
      endif
   endif
   allocate(self%u(max(n*self%ncb,1)), self%u0(max(n*self%ncb,1)))
   self%u = 0._R_P
   self%u0 = 0._R_P
   self%has_guess = .false.
   call self%make_work
   self%is_initialized = .true.
   call self%work_place(attach=.true., stat=stat)
   call agree_status(self%comm, stat, errmsg, HERE)
   if (stat /= ELISHA_OK) call self%destroy
   contains
      subroutine add_node(cc, owner, leaf)
      !< Append a node (ELISHA_ERR_INTERNAL in stat when the storage is full).
      integer(I4P), intent(in) :: cc(4) !< Block coordinates.
      integer(I4P), intent(in) :: owner !< Owner rank.
      logical,      intent(in) :: leaf  !< A leaf.

      self%nnode = self%nnode + 1
      if (self%nnode > size(self%node_owner)) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': node storage overflow')
         return
      endif
      self%node_c(:,self%nnode) = cc
      self%node_owner(self%nnode) = owner
      self%node_leaf(self%nnode) = leaf
      call self%node_hash%put(elisha_bkey(cc), self%nnode)
      endsubroutine add_node

      subroutine assemble_a
      !< The composite operator on this rank's leaf cells, row by row (`a_row`). Not symmetrised: M is symmetric by
      !< construction, so `symmetry_defect` verifies the rows, not an assembly that would force it. A row that fails
      !< (an internal invariant) stops the rows here; `assemble` agrees on the status over the ranks.
      type(triplets)            :: t        !< Assembly buffer.
      integer(I8P), allocatable :: keys(:)  !< A row: column keys.
      real(R_P),    allocatable :: vals(:)  !< A row: values.
      integer(I4P)              :: i, j, kk, e, lv !< Counters, level.
      integer(I4P)              :: cc(3)    !< Cell coordinates.

      blocks: do b=1, mesh%nb
         lv = mesh%coords(4,b)
         do kk=1, self%nijk(3)
            do j=1, self%nijk(2)
               do i=1, self%nijk(1)
                  cc = mesh%coords(1:3,b) * self%nijk + [i-1, j-1, kk-1]
                  call self%a_row(lv, cc, keys, vals, stat)
                  if (stat /= ELISHA_OK) then
                     call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': a cell neighbours a leaf '// &
                                           'two levels coarser (the mesh is not 2:1 balanced across faces)')
                     exit blocks
                  endif
                  do e=1, size(keys)
                     call trip_add(t, (b - 1) * self%ncb + self%lin([i, j, kk]), keys(e), vals(e))
                  enddo
               enddo
            enddo
         enddo
      enddo blocks
      call self%assemble(t, mesh%nb * self%ncb, SPACE_COMP, self%a, stat, errmsg)
      if (stat /= ELISHA_OK) call self%destroy
      endsubroutine assemble_a

      subroutine assemble_ghost_fill
      !< The first ghost layer of each face of every local leaf block, as leaf-cell weights.
      type(triplets)            :: t        !< Assembly buffer.
      integer(I8P), allocatable :: keys(:)  !< Value keys.
      real(R_P),    allocatable :: wts(:)   !< Value weights.
      integer(I4P), allocatable :: dst(:,:) !< Destinations.
      integer(I4P)              :: nrow, pp, dd, sgn, i, j, kk, e, a_, lv, cls !< Counters.
      integer(I4P)              :: cc(3), nb3(3), cp(3), ijk(3), f3(3) !< Coordinates.
      integer(I8P), allocatable :: keys_f(:) !< The fine cells of a coarse face.
      real(R_P)                 :: hxyz(3)   !< Cell spacing at the level.
      real(R_P)                 :: blk, e0, e1 !< A null axis: the block's extent and corners (ADAM's field%emin/emax).

      allocate(dst(4,max(2*self%nd*self%ncb*mesh%nb,1)))
      allocate(self%g_bnd(max(2*self%nd*self%ncb*mesh%nb,1)))
      allocate(self%bnd_row(max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_w(max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_x(3,max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_kind(max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_dir(max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_gs(max(2*self%nd*self%ncb*mesh%nb,1)), &
               self%bnd_gi(max(2*self%nd*self%ncb*mesh%nb,1)))
      self%g_bnd = 0
      self%bnd_gi = 0
      self%nbnd = 0
      nrow = 0
      blocks: do b=1, mesh%nb
         lv = mesh%coords(4,b)
         do pp=1, self%nd
            dd = self%act(pp)
            do sgn=-1, 1, 2
               do kk=1, self%nijk(3)
                  do j=1, self%nijk(2)
                     do i=1, self%nijk(1)
                        ijk = [i, j, kk]
                        if (sgn < 0 .and. ijk(dd) /= 1) cycle
                        if (sgn > 0 .and. ijk(dd) /= self%nijk(dd)) cycle
                        cc = mesh%coords(1:3,b) * self%nijk + ijk - 1
                        nb3 = cc ; nb3(dd) = nb3(dd) + sgn
                        nb3 = self%wrap(lv, nb3)
                        if (allocated(keys)) deallocate(keys, wts)
                        allocate(keys(0), wts(0))
                        cls = 0
                        if (self%outside(lv, nb3)) then ! a domain face: its kind's ghost, the value set by solve
                           ! (isolated: every such leaf is at lmin, elisha_check_isolated_mesh above, D-87-2)
                           self%nbnd = self%nbnd + 1
                           self%bnd_row(self%nbnd) = (b - 1) * self%ncb + self%lin(ijk)
                           self%bnd_kind(self%nbnd) = self%bc(merge(1, 2, sgn < 0),dd)
                           hxyz = self%dx_of(lv)
                           self%bnd_dir(self%nbnd) = sgn * dd
                           self%bnd_x(:,self%nbnd) = self%emin + (real(nb3, R_P) + 0.5_R_P) * self%dx_of(lv)
                           do a_=1, 3
                              if (self%null(a_)) then ! the block's centre (ADAM's field%emin/emax: the mesh's corners)
                                 blk = mesh%nijk(a_) * hxyz(a_)
                                 e0 = mesh%emin(a_) + mesh%coords(a_,b) * blk
                                 e1 = e0 + blk
                                 self%bnd_x(a_,self%nbnd) = e0 + 0.5_R_P * (e1 - e0)
                              endif
                           enddo
                           select case(self%bnd_kind(self%nbnd))
                           case(ELISHA_BC_DIRICHLET) ! phi_ghost = 2 g - phi_C, g at the face centre
                              self%bnd_w(self%nbnd) = 2._R_P * self%face_weight(lv, dd)
                              self%bnd_gs(self%nbnd) = 2._R_P
                              self%bnd_x(dd,self%nbnd) = self%bnd_x(dd,self%nbnd) - sgn * 0.5_R_P * hxyz(dd)
                              keys = [elisha_ckey(lv, cc)] ; wts = [-1._R_P]
                           case(ELISHA_BC_NEUMANN) ! phi_ghost = phi_C + h g, g the outward derivative at the face
                              self%bnd_w(self%nbnd) = self%face_weight(lv, dd) * hxyz(dd)
                              self%bnd_gs(self%nbnd) = hxyz(dd)
                              self%bnd_x(dd,self%nbnd) = self%bnd_x(dd,self%nbnd) - sgn * 0.5_R_P * hxyz(dd)
                              keys = [elisha_ckey(lv, cc)] ; wts = [1._R_P]
                           case default ! ghost-Dirichlet: phi_ghost = g, g at the ghost centre
                              self%bnd_w(self%nbnd) = self%face_weight(lv, dd)
                              self%bnd_gs(self%nbnd) = 1._R_P
                           endselect
                           self%g_bnd(nrow+1) = self%nbnd
                           ! its ghost cell in the coarsest box widened by one cell per side, i fastest
                           self%bnd_gi(self%nbnd) = (nb3(1) + 2) + (self%nc(1) + 2) * ((nb3(2) + 1) + &
                                                    (self%nc(2) + 2) * (nb3(3) + 1))
                        else
                           cls = self%classify(lv, nb3)
                           if (cls == 0) then
                              call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': a cell neighbours a '// &
                                 'leaf two levels coarser (the mesh is not 2:1 balanced across faces)')
                              exit blocks
                           endif
                        endif
                        select case(cls)
                        case(CLS_SAME)
                           keys = [elisha_ckey(lv, nb3)] ; wts = [1._R_P]
                        case(CLS_FINER) ! coarse side: phi_C + (mean_f - phi_C) / 0.75, M's face gradient over H
                           f3 = 2 * cc
                           do a_=1, self%nd
                              f3(self%act(a_)) = 2 * cc(self%act(a_))
                           enddo
                           f3(dd) = 2 * nb3(dd) + merge(0, 1, sgn > 0)
                           f3 = self%wrap(lv+1, f3)
                           call self%face_cells(lv+1, f3, dd, keys_f)
                           keys = [elisha_ckey(lv, cc)] ; wts = [1._R_P - 1._R_P / 0.75_R_P]
                           do e=1, size(keys_f)
                              keys = [keys, keys_f(e)] ; wts = [wts, 1._R_P / 0.75_R_P / size(keys_f)]
                           enddo
                        case(CLS_COARSER) ! fine side: phi_f - (mean_f - phi_C) / 1.5, M's face gradient over h
                           cp = self%parent3(nb3)
                           call self%face_cells(lv, cc, dd, keys_f)
                           keys = [elisha_ckey(lv, cc), elisha_ckey(lv-1, cp)]
                           wts = [1._R_P, 1._R_P / MG_SEAM_GHOST_SCALE]
                           do e=1, size(keys_f)
                              keys = [keys, keys_f(e)] ; wts = [wts, -1._R_P / MG_SEAM_GHOST_SCALE / size(keys_f)]
                           enddo
                        endselect
                        nrow = nrow + 1
                        ijk(dd) = ijk(dd) + sgn
                        dst(:,nrow) = [ijk, b]
                        do e=1, size(keys)
                           call trip_add(t, nrow, keys(e), wts(e))
                        enddo
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo blocks
      allocate(self%g_dst(4,max(nrow,1)))
      if (nrow > 0) self%g_dst(:,1:nrow) = dst(:,1:nrow)
      call self%assemble(t, nrow, SPACE_COMP, self%g, stat, errmsg)
      if (stat /= ELISHA_OK) call self%destroy
      endsubroutine assemble_ghost_fill

      subroutine assemble_levels
      !< The preconditioner's level operators, ghost terms and transfers, for every level above the root (lbot): the
      !< tree levels above lmin, lmin itself and the levels inside the blocks when there are some (#84 E5c).
      integer(I4P) :: lv, lt !< Level, top.

      lt = max(self%lmax, self%lbot+1)
      allocate(self%al(self%lbot+1:lt), self%el(self%lbot+1:lt), self%rl(self%lbot+1:lt), self%pl(self%lbot+1:lt), &
               self%diag(self%lbot+1:lt), self%colour(self%lbot+1:lt), self%etl(self%lbot+1:lt))
      do lv=self%lbot+1, self%lmax
         call level_ops(lv)
         if (stat /= ELISHA_OK) then ! every rank alike (assemble agreed)
            call self%destroy
            return
         endif
      enddo
      endsubroutine assemble_levels

      subroutine level_ops(lv)
      !< Level lv: A_l (the plain Laplacian on the level's cells, a face to a cell outside the level reading E_l: the
      !< level-(lv-1) cell containing it), R_l (rows: this rank's shadow cells at lv-1, the sum of their children at
      !< lv) and P_l (rows: the level's cells, their parent at lv-1). At lmin and below every cell of the domain is in
      !< the level (no E_l), and below lmin every block has its cells at lv-1 (E5c: the blocks coarsen inside). E_l^T
      !< (ELISHA notes E1c; rows: this rank's leaf cells at lv-1), from the coarse side: a leaf cell's face to a
      !< neighbour at lv-1 that is refined (its block at lv exists) has the fine cells of that face, each with the fine
      !< face weight, E_l's entries (a fine cell, the outside neighbour's parent, w) transposed. Then the red-black
      !< assertion (audit A3): every off-diagonal entry of A_l links cells of different colour.
      integer(I4P), intent(in) :: lv !< Level.
      type(triplets)           :: ta, te, tr, tp, tet !< Assembly buffers.
      integer(I4P)             :: q, nloc, i, j, kk, pp, dd, sgn, row, o, id_, ch !< Counters.
      integer(I4P)             :: cc(3), nb3(3), pc(3), cch(3) !< Coordinates.
      integer(I4P)             :: nbc(3), fc(3), m !< E_l^T: the coarse neighbour, a fine cell on the face, its bit.
      integer(I4P)             :: nl(3), nlc(3) !< Cells per block at lv and lv-1.
      real(R_P)                :: w !< Face weight.
      integer(I8P), allocatable :: hkey(:) !< A_l's halo columns: their cell keys.

      nl = self%nijk_of(lv)
      nlc = self%nijk_of(lv-1)
      nloc = self%nloc_lev(lv) * self%ncb_of(lv)
      allocate(self%colour(lv)%v(max(nloc,1)), self%diag(lv)%v(max(nloc,1)))
      self%colour(lv)%v = 0_I4P
      self%diag(lv)%v = 1._R_P
      if (nloc > 0) self%diag(lv)%v(1:nloc) = 0._R_P
      do q=1, self%nloc_lev(lv)
         id_ = self%lev_node(q,lv)
         do kk=1, nl(3)
            do j=1, nl(2)
               do i=1, nl(1)
                  cc = self%node_c(1:3,id_) * nl + [i-1, j-1, kk-1]
                  row = (q - 1) * self%ncb_of(lv) + self%lin_of(lv, [i, j, kk])
                  self%colour(lv)%v(row) = mod(sum(cc), 2)
                  do pp=1, self%nd
                     dd = self%act(pp)
                     w = self%face_weight(lv, dd)
                     do sgn=-1, 1, 2
                        nb3 = cc ; nb3(dd) = nb3(dd) + sgn
                        nb3 = self%wrap(lv, nb3)
                        if (self%outside(lv, nb3)) then ! a domain face, homogeneous: the diagonal of its kind
                           select case(self%bc(merge(1, 2, sgn < 0),dd))
                           case(ELISHA_BC_DIRICHLET)
                              call trip_add(ta, row, elisha_ckey(lv, cc), -2._R_P * w)
                              self%diag(lv)%v(row) = self%diag(lv)%v(row) - 2._R_P * w
                           case(ELISHA_BC_DIRICHLET_GHOST)
                              call trip_add(ta, row, elisha_ckey(lv, cc), -w)
                              self%diag(lv)%v(row) = self%diag(lv)%v(row) - w
                           endselect ! Neumann: no flux
                           cycle
                        endif
                        call trip_add(ta, row, elisha_ckey(lv, cc), -w)
                        self%diag(lv)%v(row) = self%diag(lv)%v(row) - w
                        if (lv <= self%lmin) then ! every cell of the domain is in the level
                           call trip_add(ta, row, elisha_ckey(lv, nb3), w)
                        elseif (self%node_hash%get(elisha_bkey([nb3 / self%nijk, lv])) > 0) then
                           call trip_add(ta, row, elisha_ckey(lv, nb3), w)
                        else
                           call trip_add(te, row, elisha_ckey(lv-1, self%parent3(nb3)), w)
                        endif
                     enddo
                  enddo
                  call trip_add(tp, row, elisha_ckey(lv-1, self%parent3(cc)), 1._R_P)
               enddo
            enddo
         enddo
      enddo
      do q=1, self%nloc_lev(lv-1)
         id_ = self%lev_node(q,lv-1)
         if (lv-1 >= self%lmin .and. self%node_leaf(id_)) cycle ! below lmin every block has its finer cells
         do kk=1, nlc(3)
            do j=1, nlc(2)
               do i=1, nlc(1)
                  pc = self%node_c(1:3,id_) * nlc + [i-1, j-1, kk-1]
                  row = (q - 1) * self%ncb_of(lv-1) + self%lin_of(lv-1, [i, j, kk])
                  do o=0, 2**self%nd-1
                     cch = pc
                     do ch=1, self%nd
                        cch(self%act(ch)) = 2 * pc(self%act(ch)) + ibits(o, ch-1, 1)
                     enddo
                     call trip_add(tr, row, elisha_ckey(lv, cch), 1._R_P)
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (lv > self%lmin) then ! E_l^T: the faces of the leaves at lv-1 to a refined neighbour
         do q=1, self%nloc_lev(lv-1)
            id_ = self%lev_node(q,lv-1)
            if (.not.self%node_leaf(id_)) cycle
            do kk=1, nlc(3)
               do j=1, nlc(2)
                  do i=1, nlc(1)
                     pc = self%node_c(1:3,id_) * nlc + [i-1, j-1, kk-1]
                     row = (q - 1) * self%ncb_of(lv-1) + self%lin_of(lv-1, [i, j, kk])
                     do pp=1, self%nd
                        dd = self%act(pp)
                        w = self%face_weight(lv, dd)
                        do sgn=-1, 1, 2
                           nbc = pc ; nbc(dd) = nbc(dd) + sgn
                           nbc = self%wrap(lv-1, nbc)
                           if (self%outside(lv-1, nbc)) cycle
                           fc = nbc
                           do ch=1, self%nd
                              fc(self%act(ch)) = 2 * nbc(self%act(ch))
                           enddo
                           if (self%node_hash%get(elisha_bkey([fc / nl, lv])) <= 0) cycle ! not refined: no seam
                           fc(dd) = fc(dd) + merge(1, 0, sgn < 0) ! the neighbour's children facing this cell
                           do o=0, 2**(self%nd-1)-1 ! the fine cells of the face: the other active axes' bits
                              cch = fc
                              m = 0
                              do ch=1, self%nd
                                 if (self%act(ch) == dd) cycle
                                 cch(self%act(ch)) = fc(self%act(ch)) + ibits(o, m, 1)
                                 m = m + 1
                              enddo
                              call trip_add(tet, row, elisha_ckey(lv, cch), w)
                           enddo
                        enddo
                     enddo
                  enddo
               enddo
            enddo
         enddo
      endif
      call self%assemble(ta, nloc, lv, self%al(lv), stat, errmsg, halo_keys=hkey) ; if (stat /= ELISHA_OK) return
      call check_colours(lv, hkey) ; if (stat /= ELISHA_OK) return
      if (self%level_operator == ELISHA_OPERATOR_STENCIL) call stencil_of(lv)
      call self%assemble(te, nloc, lv-1, self%el(lv), stat, errmsg) ; if (stat /= ELISHA_OK) return
      call self%assemble(tr, self%nloc_lev(lv-1) * self%ncb_of(lv-1), lv, self%rl(lv), stat, errmsg)
      if (stat /= ELISHA_OK) return
      call self%assemble(tp, nloc, lv-1, self%pl(lv), stat, errmsg) ; if (stat /= ELISHA_OK) return
      call self%assemble(tet, self%nloc_lev(lv-1) * self%ncb_of(lv-1), lv, self%etl(lv), stat, errmsg)
      endsubroutine level_ops

      subroutine stencil_of(lv)
      !< A_l's block-interior rows as a 7-point stencil (M7, audit C2): the blocks, their cells, the interior range per
      !< axis (2..n-1 on an active axis, 1..1 on a null one), and per active axis (act order) the row offset of the next
      !< cell and the face weight, the very w of the rows' CSR entries (`level_ops`).
      integer(I4P), intent(in) :: lv !< Level.
      integer(I4P)             :: pp, dd, nl(3) !< Counters, cells per block.

      nl = self%nijk_of(lv)
      associate(m=>self%al(lv))
      m%stencil = .true.
      m%snblk = self%nloc_lev(lv)
      m%snd = self%nd
      m%snl = nl
      m%slo = merge(1, 2, self%null) ; m%shi = merge(nl, nl - 1, self%null)
      m%soff = 0 ; m%sw = 0._R_P
      do pp=1, self%nd
         dd = self%act(pp)
         m%sw(pp) = self%face_weight(lv, dd)
         m%soff(pp) = merge(1, merge(nl(1), nl(1) * nl(2), dd == 2), dd == 1)
         ! the terms in level_ops' order: -w at the cell, w at the low neighbour, -w at the cell, w at the high one
         m%sto(4*pp-3:4*pp) = [0, -m%soff(pp), 0, m%soff(pp)]
         m%stv(4*pp-3:4*pp) = [-m%sw(pp), m%sw(pp), -m%sw(pp), m%sw(pp)]
         m%soo(2*pp-1:2*pp) = [-m%soff(pp), m%soff(pp)]
         m%sov(2*pp-1:2*pp) = [m%sw(pp), m%sw(pp)]
      enddo
      m%snt = 4 * self%nd ; m%sno = 2 * self%nd
      endassociate
      endsubroutine stencil_of

      subroutine check_colours(lv, hkey)
      !< The red-black assertion (audit A3): every off-diagonal entry of A_l (the entries `smooth` reads: a column
      !< other than the row's own cell) links cells of different colour, a halo column's colour its cell's parity.
      !< Agreed over the ranks.
      integer(I4P), intent(in) :: lv      !< Level.
      integer(I8P), intent(in) :: hkey(:) !< A_l's halo columns: their cell keys.
      integer(I4P)             :: row, k, colr, lvk !< Counters, colour, decoded level.
      integer(I4P)             :: ck(3)   !< Decoded cell.

      associate(m=>self%al(lv))
      rows: do row=1, m%nrow
         do k=m%rowptr(row), m%rowptr(row+1)-1
            if (m%col(k) == row) cycle
            if (m%col(k) <= m%nloc) then
               colr = self%colour(lv)%v(m%col(k))
            else
               call elisha_kdecode(hkey(m%col(k) - m%nloc), lvk, ck)
               colr = mod(sum(ck), 2)
            endif
            if (colr == self%colour(lv)%v(row)) then
               call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': the red-black colouring of level '// &
                  trim(str(lv, .true.))//' fails: a row couples to a cell of its own colour (the device smoother '// &
                  'updates a colour in parallel)')
               exit rows
            endif
         enddo
      enddo rows
      endassociate
      call agree_status(self%comm, stat, errmsg, HERE)
      endsubroutine check_colours
   endsubroutine setup

   subroutine solve_packed(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost, on_device)
   !< Solve from the packed interior source of this rank's leaves (block-major, `i` fastest); then the ghost fill (see
   !< `solve_packed_host`). `on_device` (M7, audit C4): `buf` and `phi` are present on the device (OpenACC), a device
   !< backend's arrays: the source is staged to a host copy (`stage_source`), `phi` to the host and back
   !< (`stage_potential`; contiguous), around the host solve, so the result is bitwise the one from host arrays. This
   !< host solver has no device arrays: `on_device=.true.` returns `ELISHA_ERR_ARGUMENT` (an argument every rank
   !< passes alike, refused before any collective). An FCG that reaches `max_iterations` above its stop tolerance
   !< returns `ELISHA_ERR_NOT_CONVERGED` on every rank (`converged` false), `phi` the last iterate with its ghosts
   !< filled (and the energy its); ADAM printed a warning on rank 0 and returned as if converged.
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh (this rank's blocks).
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                      intent(in)            :: factor  !< Source factor.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(out)           :: stat    !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg  !< Message.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.
   logical,                        intent(in),  optional :: on_device !< buf and phi on the device (default: host).
   real(R_P), allocatable :: bufh(:) !< The host copy of a device source.
   logical                :: dev     !< on_device.
   integer(I4P)           :: code    !< Status of the staging.

   dev = .false. ; if (present(on_device)) dev = on_device
   self%converged = .true.
   if (.not.dev) then
      call self%solve_packed_host(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, &
                                  energy=energy, ghost=ghost)
   else
      call self%stage_source(buf, bufh, code)
      if (code /= ELISHA_OK) then
         call elisha_set_error(stat, errmsg, code, 'elisha_hierarchical_solver_object%solve_packed: on_device needs '// &
                               'a device backend (the host solver has no device arrays)')
         return
      endif
      call self%stage_potential(phi, to_host=.true.)
      call self%solve_packed_host(mesh=mesh, buf=bufh, factor=factor, phi=phi, stat=stat, errmsg=errmsg, &
                                  energy=energy, ghost=ghost)
      if (stat == ELISHA_OK) call self%stage_potential(phi, to_host=.false.)
   endif
   ! ELISHA notes E1c, decision C: a solve that stopped at max_iterations above its tolerance is a status (every rank
   ! alike: the residual is reduced), phi the last iterate with its ghosts filled
   if (stat == ELISHA_OK .and. .not.self%converged) then
      if (self%isolated) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_CONVERGED, 'elisha_hierarchical_solver_object%'// &
            'solve_packed: the James step''s FCG solves took '//trim(str(self%iterations0, .true.))//' and '// &
            trim(str(self%iterations, .true.))//' iterations (max_iterations '//trim(str(self%max_iterations, .true.))// &
            '), the second leaving the residual at '//trim(str(self%residual))//' (stop tolerance '// &
            trim(str(self%tolerance_used))//')')
      else
         call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_CONVERGED, 'elisha_hierarchical_solver_object%'// &
            'solve_packed: the FCG reached max_iterations ('//trim(str(self%iterations, .true.))//') with the '// &
            'residual at '//trim(str(self%residual))//' (stop tolerance '//trim(str(self%tolerance_used))//')')
      endif
   endif
   endsubroutine solve_packed

   subroutine solve_packed_host(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   !< Solve from the packed interior source of this rank's leaves (block-major, `i` fastest); then the ghost fill.
   !< Isolated (issue #87, P2), the James step: the zero-ghost solve `phi0 = A^-1 factor V f`, the screening charge
   !< `s = B^T phi0` on the ghost cells, the ghost values of the free-space solution `g = -G s` (`ghost_green`), and
   !< the solve with them, `phi = A^-1 (factor V f - B g)`: the map is `(A^-1 + A^-1 B G B^T A^-1) factor V`,
   !< symmetric in the V-weighted form since A and G are. Each of the two solves warm-starts from its own last solution.
   !< Collective: the argument checks (setup done, the mesh's blocks the setup's, the shapes of buf and phi, the ghost
   !< values) are agreed over the ranks before any reduction.
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh (this rank's blocks).
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                      intent(in)            :: factor  !< Source factor.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(out)           :: stat    !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg  !< Message.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.
   real(R_P), allocatable :: g(:), s(:), u_(:) !< Ghost values, screening charge, the last solution aside.
   integer(I4P)           :: n, ub(3)          !< Interior cells of this rank, upper bounds of phi.
   logical                :: conv0             !< The zero-ghost solve of the James step converged.
   character(*), parameter :: HERE='elisha_hierarchical_solver_object%solve_packed' !< For the messages.

   if (.not.self%is_initialized) then ! every rank alike: setup is collective
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, HERE//': called before setup')
      return
   endif
   stat = ELISHA_OK
   n = mesh%nb * self%ncb
   ub = self%nijk + self%ngc
   if (size(self%wf) /= max(n,1)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_CHANGED, HERE//': the mesh does not match the setup')
   elseif (size(buf) < n) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, HERE//': the packed source is shorter than the cells')
   elseif (ubound(phi, dim=1) /= ub(1) .or. ubound(phi, dim=2) /= ub(2) .or. ubound(phi, dim=3) /= ub(3) .or. &
           size(phi, dim=4) < mesh%nb) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, HERE//': phi is not shaped [1-ngc:n+ngc per direction, '// &
                            'blocks]')
   elseif (present(ghost)) then
      call elisha_check_path(ELISHA_PATH_HIERARCHICAL, self%lmin == self%lmax, 0_I4P, self%bc, self%isolated, &
                             .false., .false., .true., stat, errmsg)
      if (stat == ELISHA_OK .and. all(self%periodic)) &
         call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_VALUES, HERE//': ghost values given on a periodic grid')
      if (stat == ELISHA_OK) call elisha_check_ghost_count(size(ghost, kind=I4P), self%nbnd, stat, errmsg)
   endif
   call agree_status(self%comm, stat, errmsg, HERE)
   if (stat /= ELISHA_OK) return
   if (.not.self%isolated) then
      call self%solve_fcg(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, energy=energy, &
                          ghost=ghost)
      return
   endif
   allocate(g(max(self%nbnd,1)), s(max(self%nbnd,1)))
   g = 0._R_P ; s = 0._R_P
   call move_alloc(self%u, u_) ; call move_alloc(self%u0, self%u)
   call self%solve_fcg(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, ghost=g)
   if (stat /= ELISHA_OK) then
      call move_alloc(self%u, self%u0) ; call move_alloc(u_, self%u)
      return
   endif
   self%iterations0 = self%iterations
   conv0 = self%converged
   call self%apply_bt(self%u, s)
   call move_alloc(self%u, self%u0) ; call move_alloc(u_, self%u)
   call self%ghost_green(s, g, stat, errmsg) ; if (stat /= ELISHA_OK) return
   call self%solve_fcg(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, energy=energy, ghost=g)
   self%converged = self%converged .and. conv0
   endsubroutine solve_packed_host

   subroutine hierarchical_solve_fcg(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   !< Solve from the packed interior source of this rank's leaves (block-major, `i` fastest): FCG on the operator,
   !< preconditioned by the V-cycle, from the previous solution when there is one; then the ghost fill. The arguments
   !< are checked (`solve_packed`).
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh.
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                      intent(in)            :: factor  !< Source factor.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(out)           :: stat    !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg  !< Message.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.
   real(R_P)              :: fmax, alpha, beta, vsum, mean, r0, e_ !< Scalars.
   real(R_P)              :: fraw(1), tol !< Max |factor f| before the mean is removed, the stop tolerance.
   real(R_P)              :: d1, d2       !< The two reductions of a ratio (ADAM's vec_dot(..) / vec_dot(..)).
   real(R_P)              :: x1(1)        !< A scalar reduced in place.
   integer(I4P)           :: n, it, b, i, j, k, c, row, l, nn !< Counters.
   logical                :: has_prev                          !< A previous direction exists.

   n = mesh%nb * self%ncb
   nn = max(n,1)
   associate(f=>self%wf)
   f = 0._R_P
   if (n > 0) f(1:n) = factor * self%vol(1:n) * buf(1:n)
   ! the boundary values to the right-hand side, f - B g; singular (no Dirichlet face): zero mean, f - V sum(f) / sum(V)
   vsum = gdot(self%comm, self%vol, [(1._R_P, i=1, nn)], n, stat) ; if (stat /= ELISHA_OK) return
   if (present(ghost)) then
      do i=1, self%nbnd
         f(self%bnd_row(i)) = f(self%bnd_row(i)) - self%bnd_w(i) * ghost(i)
      enddo
   endif
   if (self%singular) then
      mean = gdot(self%comm, f, [(1._R_P, i=1, nn)], n, stat) / vsum ; if (stat /= ELISHA_OK) return
      if (n > 0) f(1:n) = f(1:n) - mean * self%vol(1:n)
   endif
   fmax = 0._R_P
   do i=1, n
      fmax = max(fmax, abs(f(i) / self%vol(i)))
   enddo
   x1 = fmax
   call self%comm%max_real8(x1, stat) ; if (stat /= ELISHA_OK) return
   fmax = x1(1)
   ! the stop: the tolerance, or the round-off floor of the source when its mean dominates (a 1e-6 perturbation of a
   ! uniform density is known to 1e-16 / 1e-6 = 1e-10 of itself, issue #81 P5b: the iteration would never stop); the
   ! recursive residual reaches about 2 epsilon max|factor f| / max|f - mean| (measured 1.7), so the floor is 100 of it
   fraw = 0._R_P
   if (n > 0) fraw(1) = abs(factor) * maxval(abs(buf(1:n)))
   call self%comm%max_real8(fraw, stat) ; if (stat /= ELISHA_OK) return
   tol = self%tolerance
   if (fmax > 0._R_P) tol = max(tol, 100._R_P * epsilon(1._R_P) * fraw(1) / fmax)
   self%tolerance_used = tol
   endassociate
   if (.not.self%has_guess) self%u = 0._R_P
   ! the iteration on the work vectors, through the hooks (#84 E6a)
   call self%vec_update(self%wf, to_device=.true.)
   call self%vec_update(self%u, to_device=.true.)
   call self%apply(self%a, self%u, self%wq, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_lincomb(self%wf, -1._R_P, self%wq, self%wr, n)
   r0 = self%vec_relmax(self%wr, self%vol, n, fmax, stat) ; if (stat /= ELISHA_OK) return
   self%iterations = 0
   self%residual = r0
   has_prev = .false.
   do it=1, self%max_iterations
      if (self%residual <= tol .or. fmax == 0._R_P) exit
      call self%vcycle(self%wr, self%wz, stat, errmsg) ; if (stat /= ELISHA_OK) return
      if (self%singular) then
         mean = self%vec_dot(self%wz, self%vol, n, stat) / vsum ; if (stat /= ELISHA_OK) return
         call self%vec_axpy(-mean, self%w1, self%wz, n)
      endif
      call self%vec_copy(self%wz, self%wp, nn)
      if (has_prev) then
         d1 = self%vec_dot(self%wz, self%wqp, n, stat) ; if (stat /= ELISHA_OK) return
         d2 = self%vec_dot(self%wpp, self%wqp, n, stat) ; if (stat /= ELISHA_OK) return
         beta = d1 / d2
         call self%vec_lincomb(self%wz, -beta, self%wpp, self%wp, n)
      endif
      call self%apply(self%a, self%wp, self%wq, stat) ; if (stat /= ELISHA_OK) return
      d1 = self%vec_dot(self%wp, self%wr, n, stat) ; if (stat /= ELISHA_OK) return
      d2 = self%vec_dot(self%wp, self%wq, n, stat) ; if (stat /= ELISHA_OK) return
      alpha = d1 / d2
      call self%vec_axpy(alpha, self%wp, self%u, n)
      call self%vec_axpy(-alpha, self%wq, self%wr, n)
      call self%vec_copy(self%wp, self%wpp, nn)
      call self%vec_copy(self%wq, self%wqp, nn)
      has_prev = .true.
      self%iterations = it
      self%residual = self%vec_relmax(self%wr, self%vol, n, fmax, stat) ; if (stat /= ELISHA_OK) return
   enddo
   ! the residual is reduced over the ranks: every rank agrees (ADAM printed a warning on rank 0 here and went on)
   self%converged = .not.(self%residual > tol .and. fmax > 0._R_P)
   self%factor_mean = 0._R_P
   if (self%iterations > 0 .and. r0 > 0._R_P .and. self%residual > 0._R_P) &
      self%factor_mean = (self%residual / r0)**(1._R_P / self%iterations)
   ! remove the solution's mean, keep it for the next solve, fill phi
   if (self%singular) then
      mean = self%vec_dot(self%u, self%vol, n, stat) / vsum ; if (stat /= ELISHA_OK) return
      call self%vec_axpy(-mean, self%w1, self%u, n)
   endif
   call self%vec_update(self%u, to_device=.false.)
   self%has_guess = .true.
   c = 0
   do b=1, mesh%nb
      do k=1, self%nijk(3)
         do j=1, self%nijk(2)
            do i=1, self%nijk(1)
               c = c + 1
               phi(i,j,k,b) = self%u(c)
            enddo
         enddo
      enddo
   enddo
   call self%apply(self%g, self%u, self%wg, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_update(self%wg, to_device=.false.)
   associate(gv=>self%wg)
   do row=1, self%g%nrow
      b = self%g_dst(4,row)
      i = self%g_dst(1,row) ; j = self%g_dst(2,row) ; k = self%g_dst(3,row)
      if (self%g_bnd(row) > 0) then ! a domain face: the ghost of its kind (the boundary value 0 without one)
         if (self%bnd_kind(self%g_bnd(row)) == ELISHA_BC_DIRICHLET_GHOST) then
            gv(row) = 0._R_P
            if (present(ghost)) gv(row) = ghost(self%g_bnd(row))
         elseif (present(ghost)) then ! gv holds -phi_C (Dirichlet) or phi_C (Neumann)
            gv(row) = gv(row) + self%bnd_gs(self%g_bnd(row)) * ghost(self%g_bnd(row))
         endif
      endif
      phi(i,j,k,b) = gv(row)
      do l=2, self%ngc ! deeper layers copy the first along the normal
         if (i < 1)            phi(1-l,j,k,b) = gv(row)
         if (i > self%nijk(1)) phi(self%nijk(1)+l,j,k,b) = gv(row)
         if (j < 1)            phi(i,1-l,k,b) = gv(row)
         if (j > self%nijk(2)) phi(i,self%nijk(2)+l,k,b) = gv(row)
         if (k < 1)            phi(i,j,1-l,b) = gv(row)
         if (k > self%nijk(3)) phi(i,j,self%nijk(3)+l,b) = gv(row)
      enddo
   enddo
   endassociate
   if (present(energy)) then
      e_ = 0._R_P
      do c=1, n
         e_ = e_ + self%vol(c) * buf(c) * self%u(c)
      enddo
      x1 = e_
      call self%comm%sum_real8(x1, stat) ; if (stat /= ELISHA_OK) return
      energy = x1(1)
   endif
   endsubroutine hierarchical_solve_fcg

   subroutine hierarchical_solve_fcg_device(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   !< `solve_fcg` with a device backend's assembly (ELISHA notes "device assembly"): the same FCG, statement for
   !< statement, the right-hand side by the hook `assemble_rhs` and phi and the energy by `fill_potential`. A copy, not
   !< a shared loop: solve_fcg stays ADAM's code in ADAM's shape (ifx -O3 decides contraction and reduction order by
   !< it; extracting the loop or adding the hooks' calls to solve_fcg broke the bitwise parity of one case, measured).
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh.
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                      intent(in)            :: factor  !< Source factor.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(out)           :: stat    !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg  !< Message.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.
   real(R_P)              :: fmax, alpha, beta, vsum, mean, r0 !< Scalars.
   real(R_P)              :: tol          !< The stop tolerance.
   real(R_P)              :: d1, d2       !< The two reductions of a ratio (ADAM's vec_dot(..) / vec_dot(..)).
   integer(I4P)           :: n, it, nn    !< Counters.
   logical                :: has_prev                          !< A previous direction exists.

   n = mesh%nb * self%ncb
   nn = max(n,1)
   ! the right-hand side, the stop tolerance and the initial guess, by the device backend (a hook)
   call self%assemble_rhs(buf=buf, factor=factor, n=n, fmax=fmax, vsum=vsum, stat=stat, ghost=ghost)
   if (stat /= ELISHA_OK) return
   tol = self%tolerance_used
   call self%apply(self%a, self%u, self%wq, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_lincomb(self%wf, -1._R_P, self%wq, self%wr, n)
   r0 = self%vec_relmax(self%wr, self%vol, n, fmax, stat) ; if (stat /= ELISHA_OK) return
   self%iterations = 0
   self%residual = r0
   has_prev = .false.
   do it=1, self%max_iterations
      if (self%residual <= tol .or. fmax == 0._R_P) exit
      call self%vcycle(self%wr, self%wz, stat, errmsg) ; if (stat /= ELISHA_OK) return
      if (self%singular) then
         mean = self%vec_dot(self%wz, self%vol, n, stat) / vsum ; if (stat /= ELISHA_OK) return
         call self%vec_axpy(-mean, self%w1, self%wz, n)
      endif
      call self%vec_copy(self%wz, self%wp, nn)
      if (has_prev) then
         d1 = self%vec_dot(self%wz, self%wqp, n, stat) ; if (stat /= ELISHA_OK) return
         d2 = self%vec_dot(self%wpp, self%wqp, n, stat) ; if (stat /= ELISHA_OK) return
         beta = d1 / d2
         call self%vec_lincomb(self%wz, -beta, self%wpp, self%wp, n)
      endif
      call self%apply(self%a, self%wp, self%wq, stat) ; if (stat /= ELISHA_OK) return
      d1 = self%vec_dot(self%wp, self%wr, n, stat) ; if (stat /= ELISHA_OK) return
      d2 = self%vec_dot(self%wp, self%wq, n, stat) ; if (stat /= ELISHA_OK) return
      alpha = d1 / d2
      call self%vec_axpy(alpha, self%wp, self%u, n)
      call self%vec_axpy(-alpha, self%wq, self%wr, n)
      call self%vec_copy(self%wp, self%wpp, nn)
      call self%vec_copy(self%wq, self%wqp, nn)
      has_prev = .true.
      self%iterations = it
      self%residual = self%vec_relmax(self%wr, self%vol, n, fmax, stat) ; if (stat /= ELISHA_OK) return
   enddo
   ! the residual is reduced over the ranks: every rank agrees (ADAM printed a warning on rank 0 here and went on)
   self%converged = .not.(self%residual > tol .and. fmax > 0._R_P)
   self%factor_mean = 0._R_P
   if (self%iterations > 0 .and. r0 > 0._R_P .and. self%residual > 0._R_P) &
      self%factor_mean = (self%residual / r0)**(1._R_P / self%iterations)
   ! remove the solution's mean, keep it for the next solve, fill phi
   if (self%singular) then
      mean = self%vec_dot(self%u, self%vol, n, stat) / vsum ; if (stat /= ELISHA_OK) return
      call self%vec_axpy(-mean, self%w1, self%u, n)
   endif
   self%has_guess = .true.
   ! phi (the interior, the ghost fill) and the energy, by the device backend (a hook)
   call self%fill_potential(mesh=mesh, buf=buf, phi=phi, n=n, stat=stat, energy=energy, ghost=ghost)
   endsubroutine hierarchical_solve_fcg_device

   subroutine assemble_rhs(self, buf, factor, n, fmax, vsum, stat, ghost)
   !< The device assembly's hook: the right-hand side `wf`, the stop tolerance (`tolerance_used`) and the initial guess
   !< `u`, where the hooks compute. The host solver assembles inline (ADAM's code in solve_fcg): a device backend
   !< overrides it, here `ELISHA_ERR_INTERNAL`.
   class(elisha_hierarchical_solver_object), intent(inout) :: self      !< The solver.
   real(R_P),                      intent(in)            :: buf(1:)   !< Packed interior source.
   real(R_P),                      intent(in)            :: factor    !< Source factor.
   integer(I4P),                   intent(in)            :: n         !< Cells of this rank.
   real(R_P),                      intent(out)           :: fmax      !< max |f / V| over the ranks.
   real(R_P),                      intent(out)           :: vsum      !< sum V over the ranks.
   integer(I4P),                   intent(out)           :: stat      !< Status.
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.

   fmax = 0._R_P ; vsum = 0._R_P
   if (.false.) fmax = factor * buf(1) * n + ghost(1)
   stat = ELISHA_ERR_INTERNAL ! the host solver assembles inline (solve_fcg); only a device backend's is called
   endsubroutine assemble_rhs

   subroutine fill_potential(self, mesh, buf, phi, n, stat, energy, ghost)
   !< The device assembly's hook: phi from the solution `u` (the interior, the ghost fill) and the energy. The host
   !< solver fills inline (ADAM's code in solve_fcg): a device backend overrides it, here `ELISHA_ERR_INTERNAL`.
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh.
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(in)            :: n       !< Cells of this rank.
   integer(I4P),                   intent(out)           :: stat    !< Status.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.

   if (.false.) phi(1,1,1,1) = buf(1) * n * mesh%nb + ghost(1)
   if (present(energy)) energy = 0._R_P
   stat = ELISHA_ERR_INTERNAL ! the host solver fills inline (solve_fcg); only a device backend's is called
   endsubroutine fill_potential

   function symmetry_defect(self, x, y, stat) result(defect)
   !< |x . A y - y . A x| / (|x| |A y|) over the ranks, for two leaf vectors: the operator's symmetry, verified.
   !< Collective.
   class(elisha_hierarchical_solver_object), intent(inout) :: self  !< The solver.
   real(R_P),                      intent(in)    :: x(1:)        !< First vector.
   real(R_P),                      intent(in)    :: y(1:)        !< Second vector.
   integer(I4P),                   intent(out)   :: stat         !< Status.
   real(R_P)                                     :: defect       !< Result.
   real(R_P), allocatable                        :: ax(:), ay(:) !< Products.
   real(R_P)                                     :: d(4)         !< The four reductions.
   integer(I4P)                                  :: n            !< Local length.

   defect = 0._R_P
   n = self%a%nrow
   allocate(ax(max(n,1)), ay(max(n,1)))
   call self%apply_a(x, ax, stat) ; if (stat /= ELISHA_OK) return
   call self%apply_a(y, ay, stat) ; if (stat /= ELISHA_OK) return
   d(1) = gdot(self%comm, x, ay, n, stat) ; if (stat /= ELISHA_OK) return
   d(2) = gdot(self%comm, y, ax, n, stat) ; if (stat /= ELISHA_OK) return
   d(3) = gdot(self%comm, x, x, n, stat) ; if (stat /= ELISHA_OK) return
   d(4) = gdot(self%comm, ay, ay, n, stat) ; if (stat /= ELISHA_OK) return
   defect = abs(d(1) - d(2)) / (sqrt(d(3)) * sqrt(d(4)))
   endfunction symmetry_defect

   function vcycle_symmetry_defect(self, x, y, stat) result(defect)
   !< |x . M y - y . M x| / (|x| |M y|) over the ranks, M the V-cycle (the FCG preconditioner), for two leaf vectors:
   !< the preconditioner's symmetry, verified (ELISHA notes E1c: about 1e-16 with `ELISHA_VCYCLE_SYMMETRIC`, O(0.1)
   !< with `ELISHA_VCYCLE_ADAM` on AMR, about 1e-16 on a uniform mesh with either). Through the FCG work vectors (the
   !< next solve recomputes them). Collective.
   class(elisha_hierarchical_solver_object), intent(inout) :: self  !< The solver.
   real(R_P),                      intent(in)    :: x(1:)        !< First vector.
   real(R_P),                      intent(in)    :: y(1:)        !< Second vector.
   integer(I4P),                   intent(out)   :: stat         !< Status.
   real(R_P)                                     :: defect       !< Result.
   real(R_P), allocatable                        :: mx(:), my(:) !< Products.
   real(R_P)                                     :: d(4)         !< The four reductions.
   integer(I4P)                                  :: n            !< Local length.

   defect = 0._R_P
   if (.not.self%is_initialized) then
      stat = ELISHA_ERR_NOT_INITIALIZED
      return
   endif
   n = self%a%nrow
   allocate(mx(max(n,1)), my(max(n,1)))
   ! M x and M y through the work vectors (inline: nvfortran 26.9 refuses a type-bound call on the host's self from an
   ! internal procedure)
   if (n > 0) self%wr(1:n) = x(1:n)
   call self%vec_update(self%wr, to_device=.true.)
   call self%vcycle(self%wr, self%wz, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_update(self%wz, to_device=.false.)
   if (n > 0) mx(1:n) = self%wz(1:n)
   if (n > 0) self%wr(1:n) = y(1:n)
   call self%vec_update(self%wr, to_device=.true.)
   call self%vcycle(self%wr, self%wz, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_update(self%wz, to_device=.false.)
   if (n > 0) my(1:n) = self%wz(1:n)
   d(1) = gdot(self%comm, x, my, n, stat) ; if (stat /= ELISHA_OK) return
   d(2) = gdot(self%comm, y, mx, n, stat) ; if (stat /= ELISHA_OK) return
   d(3) = gdot(self%comm, x, x, n, stat) ; if (stat /= ELISHA_OK) return
   d(4) = gdot(self%comm, my, my, n, stat) ; if (stat /= ELISHA_OK) return
   defect = abs(d(1) - d(2)) / (sqrt(d(3)) * sqrt(d(4)))
   endfunction vcycle_symmetry_defect

   subroutine apply_a(self, x, y, stat)
   !< y = A x on this rank's leaf cells (V-weighted), through the work vectors (where the hooks compute). Collective.
   class(elisha_hierarchical_solver_object), intent(inout) :: self  !< The solver.
   real(R_P),                      intent(in)    :: x(1:) !< Leaf vector.
   real(R_P),                      intent(inout) :: y(1:) !< Result.
   integer(I4P),                   intent(out)   :: stat  !< Status.
   integer(I4P)                                  :: n     !< Rows.

   n = self%a%nrow
   if (n > 0) self%wp(1:n) = x(1:n)
   call self%vec_update(self%wp, to_device=.true.)
   call self%apply(self%a, self%wp, self%wq, stat) ; if (stat /= ELISHA_OK) return
   call self%vec_update(self%wq, to_device=.false.)
   if (n > 0) y(1:n) = self%wq(1:n)
   endsubroutine apply_a

   subroutine apply_b(self, g, y)
   !< y = B g on this rank's leaf cells: every boundary face adds its weight times its boundary value to its row
   !< (`A phi + B g` is the operator with the boundary values g).
   class(elisha_hierarchical_solver_object), intent(in)    :: self  !< The solver.
   real(R_P),                      intent(in)    :: g(1:) !< Boundary value per boundary face.
   real(R_P),                      intent(inout) :: y(1:) !< Result, a leaf vector.
   integer(I4P)                                  :: i     !< Counter.

   y(1:self%a%nrow) = 0._R_P
   do i=1, self%nbnd
      y(self%bnd_row(i)) = y(self%bnd_row(i)) + self%bnd_w(i) * g(i)
   enddo
   endsubroutine apply_b

   subroutine apply_bt(self, x, s)
   !< s = B^T x: every boundary face takes its weight times the value of its row.
   class(elisha_hierarchical_solver_object), intent(in)    :: self  !< The solver.
   real(R_P),                      intent(in)    :: x(1:) !< Leaf vector.
   real(R_P),                      intent(inout) :: s(1:) !< Result, one value per boundary face.
   integer(I4P)                                  :: i     !< Counter.

   do i=1, self%nbnd
      s(i) = self%bnd_w(i) * x(self%bnd_row(i))
   enddo
   endsubroutine apply_bt

   ! backend hooks
   subroutine apply(self, m, x, y, stat)
   !< y = M x: fetch the halo of x from the owners, then the local rows. The exchange buffers are the solver's
   !< (`xsend`, `xhalo`, allocated once by `make_work`, audit C1); only their first `size(m%sidx)` and `m%nhalo`
   !< entries are written and read.
   class(elisha_hierarchical_solver_object), intent(inout) :: self   !< The solver.
   type(dcsr),                     intent(in)    :: m      !< Operator.
   real(R_P),                      intent(in)    :: x(1:)  !< Local column-space vector.
   real(R_P),                      intent(inout) :: y(1:)  !< Result (rows).
   integer(I4P),                   intent(out)   :: stat   !< Status.
   integer(I4P)                                  :: i, k, p !< Counters.
   integer(I4P)                                  :: q, ii, jj, kk !< Block, cell (stencil rows).
   real(R_P)                                     :: acc    !< Accumulator.

   associate(sbuf=>self%xsend, halo=>self%xhalo)
   do i=1, size(m%sidx)
      sbuf(i) = x(m%sidx(i))
   enddo
   call m%plan%exchange_r(sbuf, halo, stat)
   if (stat /= ELISHA_OK) return
   if (.not.m%stencil) then ! ADAM's loop, unchanged
      do i=1, m%nrow
         acc = 0._R_P
         do k=m%rowptr(i), m%rowptr(i+1)-1
            if (m%col(k) <= m%nloc) then
               acc = acc + m%val(k) * x(m%col(k))
            else
               acc = acc + m%val(k) * halo(m%col(k) - m%nloc)
            endif
         enddo
         y(i) = acc
      enddo
   else ! the block-interior rows as the stencil, in their CSR entries' order (M7, audit C2)
      do q=1, m%snblk
         do kk=1, m%snl(3)
            do jj=1, m%snl(2)
               do ii=1, m%snl(1)
                  i = (q - 1) * m%snl(1) * m%snl(2) * m%snl(3) + ii + (jj - 1) * m%snl(1) + (kk - 1) * m%snl(1) * m%snl(2)
                  if (ii < m%slo(1) .or. ii > m%shi(1) .or. jj < m%slo(2) .or. jj > m%shi(2) .or. &
                      kk < m%slo(3) .or. kk > m%shi(3)) then
                     y(i) = csr_row(i)
                  else
                     acc = 0._R_P
                     do p=1, m%snt
                        acc = acc + m%stv(p) * x(i + m%sto(p))
                     enddo
                     y(i) = acc
                  endif
               enddo
            enddo
         enddo
      enddo
   endif
   endassociate
   contains
      function csr_row(r) result(a)
      !< Row r of m x from its CSR entries.
      integer(I4P), intent(in) :: r  !< Row.
      real(R_P)                :: a  !< Its value.
      integer(I4P)             :: k_ !< Counter.

      a = 0._R_P
      do k_=m%rowptr(r), m%rowptr(r+1)-1
         if (m%col(k_) <= m%nloc) then
            a = a + m%val(k_) * x(m%col(k_))
         else
            a = a + m%val(k_) * self%xhalo(m%col(k_) - m%nloc)
         endif
      enddo
      endfunction csr_row
   endsubroutine apply

   subroutine vec_fill(x, a, n)
   !< x(1:n) = a.
   real(R_P),    intent(inout) :: x(1:) !< Vector.
   real(R_P),    intent(in)    :: a     !< Value.
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) x(1:n) = a
   endsubroutine vec_fill

   subroutine vec_copy(x, y, n)
   !< y(1:n) = x(1:n).
   real(R_P),    intent(in)    :: x(1:) !< Source.
   real(R_P),    intent(inout) :: y(1:) !< Destination.
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) y(1:n) = x(1:n)
   endsubroutine vec_copy

   subroutine vec_lincomb(x, a, y, w, n)
   !< w(1:n) = x(1:n) + a y(1:n) (w distinct from x and y).
   real(R_P),    intent(in)    :: x(1:) !< First vector.
   real(R_P),    intent(in)    :: a     !< Coefficient.
   real(R_P),    intent(in)    :: y(1:) !< Second vector.
   real(R_P),    intent(inout) :: w(1:) !< Result.
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) w(1:n) = x(1:n) + a * y(1:n)
   endsubroutine vec_lincomb

   subroutine vec_axpy(a, x, y, n)
   !< y(1:n) = y(1:n) + a x(1:n).
   real(R_P),    intent(in)    :: a     !< Coefficient.
   real(R_P),    intent(in)    :: x(1:) !< Vector.
   real(R_P),    intent(inout) :: y(1:) !< Updated vector.
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) y(1:n) = y(1:n) + a * x(1:n)
   endsubroutine vec_axpy

   function vec_dot(self, x, y, n, stat) result(d)
   !< sum_i x_i y_i over the local n entries and the ranks.
   class(elisha_hierarchical_solver_object), intent(in) :: self  !< The solver (its communicator).
   real(R_P),                      intent(in)  :: x(1:) !< First vector.
   real(R_P),                      intent(in)  :: y(1:) !< Second vector.
   integer(I4P),                   intent(in)  :: n     !< Local entries.
   integer(I4P),                   intent(out) :: stat  !< Status.
   real(R_P)                                   :: d     !< Result.

   d = gdot(self%comm, x, y, n, stat)
   endfunction vec_dot

   function vec_relmax(self, r, vol, n, fmax, stat) result(rm)
   !< max |r / vol| over the ranks, over fmax (when positive).
   class(elisha_hierarchical_solver_object), intent(in) :: self    !< The solver (its communicator).
   real(R_P),                      intent(in)  :: r(1:)   !< Residual.
   real(R_P),                      intent(in)  :: vol(1:) !< Cell volumes.
   integer(I4P),                   intent(in)  :: n       !< Local entries.
   real(R_P),                      intent(in)  :: fmax    !< Normalisation.
   integer(I4P),                   intent(out) :: stat    !< Status.
   real(R_P)                                   :: rm      !< Result.
   real(R_P)                                   :: x1(1)   !< rm, reduced in place.
   integer(I4P)                                :: i       !< Counter.

   rm = 0._R_P
   do i=1, n
      rm = max(rm, abs(r(i) / vol(i)))
   enddo
   x1 = rm
   call self%comm%max_real8(x1, stat)
   rm = x1(1)
   if (fmax > 0._R_P) rm = rm / fmax
   endfunction vec_relmax

   subroutine vec_permute(x, ix, y, iy, n)
   !< y(iy(i)) = x(ix(i)), i = 1..n.
   real(R_P),    intent(in)    :: x(1:)  !< Source.
   integer(I4P), intent(in)    :: ix(1:) !< Source entries.
   real(R_P),    intent(inout) :: y(1:)  !< Destination.
   integer(I4P), intent(in)    :: iy(1:) !< Destination entries.
   integer(I4P), intent(in)    :: n      !< Entries.
   integer(I4P)                :: i      !< Counter.

   do i=1, n
      y(iy(i)) = x(ix(i))
   enddo
   endsubroutine vec_permute

   subroutine root_solve(self, n, nc, v0, stat, errmsg)
   !< The V-cycle's root (lbot): e = L^-1 (r / V), the spectral solver on its `nc` cells (`n` blocks of this rank), every
   !< rank (the mean removed), through the host: r down from where the hooks compute, e back up (ADAM's path). A device
   !< backend overrides it to keep the root on the device (M7, audit C4).
   class(elisha_hierarchical_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                             intent(in)            :: n      !< Root blocks of this rank.
   integer(I4P),                             intent(in)            :: nc     !< Root cells of this rank.
   real(R_P),                                intent(in)            :: v0     !< Root cell volume.
   integer(I4P),                             intent(out)           :: stat   !< Status.
   character(:), allocatable,                intent(out), optional :: errmsg !< Message.

   call self%vec_update(self%lr(self%lbot)%v, to_device=.false.)
   self%root_buf = 0._R_P
   if (nc > 0) self%root_buf(1:nc) = self%lr(self%lbot)%v(1:nc) / v0
   call self%fft%solve_blocks(nbl=n, coords=self%root_coords, buf=self%root_buf, factor=1._R_P, gw=0_I4P, &
                              out=self%root_out, stat=stat, errmsg=errmsg)
   if (stat /= ELISHA_OK) return
   if (nc > 0) self%le(self%lbot)%v(1:nc) = self%root_out(1:nc)
   call self%vec_update(self%le(self%lbot)%v, to_device=.true.)
   endsubroutine root_solve

   subroutine stage_source(self, buf, bufh, stat)
   !< `solve_packed(on_device)`: a host copy of the device source. The host solver has no device arrays:
   !< `ELISHA_ERR_ARGUMENT` (a device backend overrides it).
   class(elisha_hierarchical_solver_object), intent(inout) :: self    !< The solver.
   real(R_P),                                intent(in)    :: buf(1:) !< Source (on the device).
   real(R_P), allocatable,                   intent(out)   :: bufh(:) !< Its host copy.
   integer(I4P),                             intent(out)   :: stat    !< Status.

   if (.false.) bufh = buf ! the host solver: unused
   stat = ELISHA_ERR_ARGUMENT
   endsubroutine stage_source

   subroutine stage_potential(self, phi, to_host)
   !< `solve_packed(on_device)`: phi from the device to the host (`to_host`) or back. Nothing here (the host solver
   !< refuses `on_device` in `stage_source`).
   class(elisha_hierarchical_solver_object), intent(inout) :: self          !< The solver.
   real(R_P),                                intent(inout) :: phi(:,:,:,:)  !< Potential (contiguous).
   logical,                                  intent(in)    :: to_host       !< Direction.

   if (.false.) phi(1,1,1,1) = merge(1._R_P, 0._R_P, to_host)
   endsubroutine stage_potential

   subroutine vec_update(x, to_device)
   !< A work vector to (`to_device`) or from where the hooks compute: nothing on the host.
   real(R_P), intent(inout) :: x(1:)     !< Work vector.
   logical,   intent(in)    :: to_device !< Direction.

   if (to_device .or. size(x) == 0) return
   endsubroutine vec_update

   subroutine work_place(self, attach, stat)
   !< Put (`attach`) the operators and the work vectors where the hooks compute, or release them: nothing on the host.
   class(elisha_hierarchical_solver_object), intent(inout) :: self   !< The solver.
   logical,                        intent(in)    :: attach !< Put, or release.
   integer(I4P),                   intent(out)   :: stat   !< Status.

   if (.not.self%is_initialized) then
      stat = ELISHA_ERR_NOT_INITIALIZED
      return
   endif
   stat = ELISHA_OK
   if (attach) return
   endsubroutine work_place

   ! private methods
   subroutine assemble(self, t, nrow, space, m, stat, errmsg, halo_keys)
   !< Finish a distributed operator: rows sorted; column keys resolved to local slots, or to halo slots grouped by
   !< owner; the exchange plan agreed with the owners (they receive the keys and return their local indices). `stat`
   !< enters with the status of the triplets' construction on this rank: the statuses are agreed over the ranks before
   !< the exchanges (a failed rank skips its keys) and after them. `halo_keys`: the cell key of each halo column.
   class(elisha_hierarchical_solver_object), intent(in)    :: self  !< The solver.
   type(triplets),                 intent(inout) :: t     !< Triplets (emptied).
   integer(I4P),                   intent(in)    :: nrow  !< Rows.
   integer(I4P),                   intent(in)    :: space !< Column space.
   type(dcsr),                     intent(inout) :: m     !< Operator.
   integer(I4P),                   intent(inout) :: stat  !< Status (this rank's on entry; agreed on exit).
   character(:), allocatable,      intent(inout), optional :: errmsg !< Message.
   integer(I8P), allocatable,      intent(out),   optional :: halo_keys(:) !< Cell key per halo column [nhalo].
   type(elisha_hash_table)   :: hh                  !< Halo key -> provisional halo id.
   integer(I8P), allocatable :: hkey(:), ordered(:), skey(:) !< Halo keys: provisional, by slot, requested.
   integer(I4P), allocatable :: hown(:), slot(:), prov(:), fill(:), next(:) !< Owners, slots, column ids, cursors.
   integer(I4P)              :: e, own, lidx, nh, i, r, k, nsend !< Counters.
   character(*), parameter   :: HERE='elisha_hierarchical_solver_object%assemble' !< For the messages.

   m%nrow = nrow
   if (space == SPACE_COMP) then
      m%nloc = self%nleaf_loc * self%ncb
   else
      m%nloc = self%nloc_lev(space) * self%ncb_of(space)
   endif
   allocate(prov(max(t%n,1)), hkey(max(t%n,1)), hown(max(t%n,1)))
   call hh%init(max(t%n,1))
   nh = 0
   if (stat == ELISHA_OK) then
      do e=1, t%n
         call self%locate(space, t%k(e), own, lidx)
         if (own < 0) then
            call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': a column key names no leaf or node '// &
                                  'of its space')
            exit
         endif
         if (own == self%me) then
            prov(e) = lidx
         else
            i = hh%get(t%k(e))
            if (i == 0) then
               nh = nh + 1
               hkey(nh) = t%k(e)
               hown(nh) = own
               call hh%put(t%k(e), nh)
               i = nh
            endif
            prov(e) = -i
         endif
      enddo
   endif
   call hh%free
   call agree_status(self%comm, stat, errmsg, HERE) ; if (stat /= ELISHA_OK) return
   ! halo slots grouped by owner
   m%nhalo = nh
   allocate(m%rcount(0:self%nr-1), m%rdispl(0:self%nr-1), fill(0:self%nr-1), slot(max(nh,1)), ordered(max(nh,1)))
   m%rcount = 0
   do i=1, nh
      m%rcount(hown(i)) = m%rcount(hown(i)) + 1
   enddo
   m%rdispl(0) = 0
   do r=1, self%nr-1
      m%rdispl(r) = m%rdispl(r-1) + m%rcount(r-1)
   enddo
   fill = m%rdispl
   do i=1, nh
      fill(hown(i)) = fill(hown(i)) + 1
      slot(i) = fill(hown(i))
      ordered(slot(i)) = hkey(i)
   enddo
   if (present(halo_keys)) then
      allocate(halo_keys(max(nh,1)))
      if (nh > 0) halo_keys(1:nh) = ordered(1:nh)
   endif
   ! the requests: the owners translate the keys into their local indices
   allocate(m%scount(0:self%nr-1), m%sdispl(0:self%nr-1))
   call self%comm%alltoall_i4(m%rcount, m%scount, stat) ; if (stat /= ELISHA_OK) return
   m%sdispl(0) = 0
   do r=1, self%nr-1
      m%sdispl(r) = m%sdispl(r-1) + m%scount(r-1)
   enddo
   nsend = sum(m%scount)
   allocate(skey(max(nsend,1)), m%sidx(nsend))
   call self%comm%alltoallv_i8(ordered, m%rcount, m%rdispl, skey, m%scount, m%sdispl, stat)
   if (stat /= ELISHA_OK) return
   do i=1, nsend
      call self%locate(space, skey(i), own, lidx)
      if (own /= self%me) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_INTERNAL, HERE//': a halo request reached the wrong owner')
         exit
      endif
      m%sidx(i) = lidx
   enddo
   call agree_status(self%comm, stat, errmsg, HERE) ; if (stat /= ELISHA_OK) return
   ! the neighbour exchange of the plan (M7, audit B1): collective, every rank reaches it
   call m%plan%init(self%comm, m%scount, m%sdispl, m%rcount, m%rdispl, i)
   if (i /= ELISHA_OK) call elisha_set_error(stat, errmsg, i, HERE//': the halo plan''s graph communicator')
   call agree_status(self%comm, stat, errmsg, HERE)
   if (stat /= ELISHA_OK) return
   ! rows: counting sort of the triplets
   allocate(m%rowptr(nrow+1), m%col(max(t%n,1)), m%val(max(t%n,1)), next(max(nrow,1)))
   m%rowptr = 0
   do e=1, t%n
      m%rowptr(t%r(e)+1) = m%rowptr(t%r(e)+1) + 1
   enddo
   m%rowptr(1) = 1
   do i=1, nrow
      m%rowptr(i+1) = m%rowptr(i+1) + m%rowptr(i)
   enddo
   if (nrow > 0) next(1:nrow) = m%rowptr(1:nrow)
   do e=1, t%n
      k = next(t%r(e))
      next(t%r(e)) = k + 1
      if (prov(e) > 0) then
         m%col(k) = prov(e)
      else
         m%col(k) = m%nloc + slot(-prov(e))
      endif
      m%val(k) = t%v(e)
   enddo
   t%n = 0
   endsubroutine assemble

   subroutine a_row(self, lv, cc, keys, vals, stat)
   !< The row of the composite operator at cell cc of level lv (V-weighted): same-level faces, and the fine and the
   !< coarse side of 2:1 faces (see the module). ELISHA_ERR_INTERNAL when a neighbour cannot be classified.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),              intent(in)    :: lv      !< Level.
   integer(I4P),              intent(in)    :: cc(3)   !< Cell coordinates at lv.
   integer(I8P), allocatable, intent(inout) :: keys(:) !< Column keys.
   real(R_P),    allocatable, intent(inout) :: vals(:) !< Values.
   integer(I4P),              intent(out)   :: stat    !< Status.
   integer(I4P) :: p, d, sgn, cls, o, ch, nf          !< Counters.
   integer(I4P) :: nb3(3), cp(3), f3(3)               !< Coordinates.
   integer(I8P) :: ky                                  !< This cell's key.
   integer(I8P), allocatable :: keys_f(:)              !< The fine cells of a coarse face.
   real(R_P)    :: w, wf                               !< Weights.

   stat = ELISHA_OK
   if (allocated(keys)) deallocate(keys, vals)
   allocate(keys(0), vals(0))
   ky = elisha_ckey(lv, cc)
   do p=1, self%nd
      d = self%act(p)
      do sgn=-1, 1, 2
         nb3 = cc ; nb3(d) = nb3(d) + sgn
         nb3 = self%wrap(lv, nb3)
         if (self%outside(lv, nb3)) then ! a domain face: the homogeneous row of its kind, the value to the right side
            w = self%face_weight(lv, d)
            select case(self%bc(merge(1, 2, sgn < 0),d))
            case(ELISHA_BC_DIRICHLET)
               keys = [keys, ky] ; vals = [vals, -2._R_P * w]
            case(ELISHA_BC_DIRICHLET_GHOST)
               keys = [keys, ky] ; vals = [vals, -w]
            endselect ! Neumann: no flux
            cycle
         endif
         cls = self%classify(lv, nb3)
         select case(cls)
         case(CLS_SAME)
            w = self%face_weight(lv, d)
            keys = [keys, ky, elisha_ckey(lv, nb3)] ; vals = [vals, -w, w]
         case(CLS_COARSER) ! the fine side: the mimetic mean flux (2/3) w (mean of the face's fine cells - phi_C)
            w = self%face_weight(lv, d)
            cp = self%parent3(nb3)
            call self%face_cells(lv, cc, d, keys_f)
            nf = size(keys_f)
            do o=1, nf
               keys = [keys, keys_f(o)] ; vals = [vals, -2._R_P / 3._R_P * w / nf]
            enddo
            keys = [keys, elisha_ckey(lv-1, cp)] ; vals = [vals, 2._R_P / 3._R_P * w]
         case(CLS_FINER) ! the coarse side: the adjoint, (2/3) w_f (sum of the face's fine cells - n phi_C)
            wf = self%face_weight(lv+1, d)
            f3 = 2 * cc
            do ch=1, self%nd
               f3(self%act(ch)) = 2 * cc(self%act(ch))
            enddo
            f3(d) = 2 * nb3(d) + merge(0, 1, sgn > 0) ! the children of the neighbour touching this face
            f3 = self%wrap(lv+1, f3)
            call self%face_cells(lv+1, f3, d, keys_f)
            nf = size(keys_f)
            do o=1, nf
               keys = [keys, keys_f(o), ky] ; vals = [vals, 2._R_P / 3._R_P * wf, -2._R_P / 3._R_P * wf]
            enddo
         case default ! a leaf two levels coarser (validate refuses such a mesh)
            stat = ELISHA_ERR_INTERNAL
            return
         endselect
      enddo
   enddo
   endsubroutine a_row

   subroutine face_cells(self, lf, fc, d, keys_f)
   !< The keys of the 2^(nd-1) fine cells (level lf) on the coarse face that contains the fine cell fc's face normal
   !< to d: fc's normal index, every pair of tangential indices of its coarse parent.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),              intent(in)  :: lf        !< Fine level.
   integer(I4P),              intent(in)  :: fc(3)     !< A fine cell of the face.
   integer(I4P),              intent(in)  :: d         !< Normal direction.
   integer(I8P), allocatable, intent(out) :: keys_f(:) !< Keys.
   integer(I4P)                           :: o, ch, a_, nt !< Counters.
   integer(I4P)                           :: c2(3), tang(2) !< Coordinates, tangential directions.

   nt = 0
   do ch=1, self%nd
      if (self%act(ch) == d) cycle
      nt = nt + 1
      tang(nt) = self%act(ch)
   enddo
   allocate(keys_f(2**nt))
   do o=0, 2**nt-1
      c2 = fc
      do a_=1, nt
         c2(tang(a_)) = 2 * (fc(tang(a_)) / 2) + ibits(o, a_-1, 1)
      enddo
      keys_f(o+1) = elisha_ckey(lf, c2)
   enddo
   endsubroutine face_cells

   subroutine ghost_green(self, s, g, stat, errmsg)
   !< Isolated: `g = -G s` on this rank's boundary faces, `G` the inverse of the V-weighted coarsest-level operator on
   !< the infinite lattice, from ghost cell to ghost cell (`G(n) / H`, the unit lattice's Green's function over the
   !< coarsest spacing). Rank 0 gathers the charges into the coarsest box widened by the ghost layer (every face has
   !< its own ghost cell: a boundary leaf is at the coarsest level), convolves with the padded FFT (`gfft`, which solves
   !< `L_H phi = factor f` by the kernel `H^2 G`, so `factor = -1 / H^3`), and scatters the values back.
   class(elisha_hierarchical_solver_object), intent(inout) :: self  !< The solver.
   real(R_P),                      intent(in)    :: s(1:) !< Screening charge per boundary face.
   real(R_P),                      intent(inout) :: g(1:) !< Ghost value per boundary face.
   integer(I4P),                   intent(out)   :: stat  !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg !< Message.
   integer(I4P), allocatable :: counts(:), displs(:) !< Gather counts and displacements.
   integer(I4P), allocatable :: gi_all(:)            !< Rank 0: the ghost cells of every face.
   real(R_P),    allocatable :: v_all(:)             !< Rank 0: the charges, then the values, of every face.
   real(R_P),    allocatable :: box(:), out(:)       !< Rank 0: the widened box, source and potential.
   real(R_P)                 :: h(3)                 !< Coarsest spacing.
   integer(I4P)              :: ntot, nbl, c, r, st  !< Faces, blocks of the box solve, counters, a failed status.
   character(*), parameter   :: HERE='elisha_hierarchical_solver_object%ghost_green' !< For the messages.

   allocate(counts(self%nr), displs(self%nr))
   counts = 0
   call self%comm%gather_i4(self%nbnd, counts, stat)
   if (stat /= ELISHA_OK) then
      st = stat
      call elisha_set_error(stat, errmsg, st, HERE//': a collective failed')
      return
   endif
   displs = 0 ; ntot = 0
   if (self%me == 0) then
      do r=2, self%nr
         displs(r) = displs(r-1) + counts(r-1)
      enddo
      ntot = sum(counts)
   endif
   allocate(gi_all(max(ntot,1)), v_all(max(ntot,1)))
   call self%comm%gatherv_i4(self%bnd_gi, self%nbnd, gi_all, counts, displs, stat)
   if (stat == ELISHA_OK) call self%comm%gatherv_r(s, self%nbnd, v_all, counts, displs, stat)
   if (stat /= ELISHA_OK) then
      st = stat
      call elisha_set_error(stat, errmsg, st, HERE//': a collective failed')
      return
   endif
   nbl = 0
   if (self%me == 0) then
      nbl = 1
      allocate(box(product(self%nc + 2)), out(product(self%nc + 2)))
      box = 0._R_P
      do c=1, ntot
         box(gi_all(c)) = box(gi_all(c)) + v_all(c)
      enddo
   else
      allocate(box(1), out(1))
   endif
   h = self%dx_of(self%lmin)
   call self%gfft%solve_blocks(nbl=nbl, coords=reshape([0_I4P, 0_I4P, 0_I4P], [3, 1]), buf=box, &
                               factor=-1._R_P / h(1)**3, gw=0_I4P, out=out, stat=stat, errmsg=errmsg)
   if (stat /= ELISHA_OK) return ! every rank alike (the spectral solver agreed)
   if (self%me == 0) then
      do c=1, ntot
         v_all(c) = out(gi_all(c))
      enddo
   endif
   call self%comm%scatterv_r(v_all, counts, displs, g, self%nbnd, stat)
   if (stat /= ELISHA_OK) then
      st = stat
      call elisha_set_error(stat, errmsg, st, HERE//': a collective failed')
   endif
   endsubroutine ghost_green

   pure function cell_volume(self, lv) result(v)
   !< Cell volume at level lv (the product over the active directions).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv   !< Level.
   real(R_P)                            :: v    !< Volume.
   real(R_P)                            :: dx(3) !< Spacing.
   integer(I4P)                         :: p    !< Counter.

   dx = self%dx_of(lv)
   v = 1._R_P
   do p=1, self%nd
      v = v * dx(self%act(p))
   enddo
   endfunction cell_volume

   pure function classify(self, lv, nb3) result(cls)
   !< Classify the level-lv cell position nb3: a leaf of lv, refined, or inside a leaf of lv-1; 0 otherwise (the mesh
   !< is not 2:1 balanced across faces: an internal invariant, `validate` refuses such a mesh).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv     !< Level.
   integer(I4P),             intent(in) :: nb3(3) !< Cell coordinates.
   integer(I4P)                         :: cls    !< Class.
   integer(I4P)                         :: id     !< Node id.

   id = self%node_hash%get(elisha_bkey([nb3 / self%nijk, lv]))
   if (id > 0) then
      cls = merge(CLS_SAME, CLS_FINER, self%node_leaf(id))
      return
   endif
   if (lv > self%lmin) then
      if (self%leaf_hash%get(elisha_bkey([self%parent3(nb3) / self%nijk, lv-1])) > 0) then
         cls = CLS_COARSER
         return
      endif
   endif
   cls = 0
   endfunction classify

   pure function dx_of(self, lv) result(dx)
   !< Cell spacing at level lv (null directions keep the level-0 one).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv    !< Level.
   real(R_P)                            :: dx(3) !< Spacing.

   dx = merge(self%dx0, self%dx0 / 2._R_P**lv, self%null)
   endfunction dx_of

   pure function face_weight(self, lv, d) result(w)
   !< The V-weighted face weight at level lv along d: the face area over the centre distance, V / dx_d^2.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv   !< Level.
   integer(I4P),             intent(in) :: d    !< Direction.
   real(R_P)                            :: w    !< Weight.
   real(R_P)                            :: dx(3) !< Spacing.

   dx = self%dx_of(lv)
   w = self%cell_volume(lv) / dx(d)**2
   endfunction face_weight

   subroutine make_work(self)
   !< Allocate the work vectors (FCG, ghost fill, the V-cycle's levels and root), the maps of the leaves into the
   !< level spaces (#84 E6a) and the halo-exchange buffers of `apply` and `smooth` (audit C1: the largest send and
   !< halo over the operators, at least 1).
   class(elisha_hierarchical_solver_object), intent(inout) :: self !< The solver.
   integer(I4P)                            :: nn, lv, b, c, i, q, n !< Sizes, counters.
   integer(I4P)                            :: ns, nh !< Largest send and halo.

   nn = size(self%u)
   allocate(self%wf(nn), self%wr(nn), self%wz(nn), self%wp(nn), self%wq(nn), self%wpp(nn), self%wqp(nn), self%w1(nn), &
            self%wg(max(self%g%nrow,1)))
   self%wf = 0._R_P ; self%wr = 0._R_P ; self%wz = 0._R_P ; self%wp = 0._R_P ; self%wq = 0._R_P
   self%wpp = 0._R_P ; self%wqp = 0._R_P ; self%w1 = 1._R_P ; self%wg = 0._R_P
   allocate(self%lr(self%lbot:self%lmax), self%le(self%lbot:self%lmax), self%lt1(self%lbot:self%lmax), &
            self%lt2(self%lbot:self%lmax), self%mlev(self%lbot:self%lmax), self%mleaf(self%lbot:self%lmax))
   do lv=self%lbot, self%lmax
      n = max(self%nloc_lev(lv) * self%ncb_of(lv), 1)
      allocate(self%lr(lv)%v(n), self%le(lv)%v(n), self%lt1(lv)%v(n), self%lt2(lv)%v(n))
      self%lr(lv)%v = 0._R_P ; self%le(lv)%v = 0._R_P ; self%lt1(lv)%v = 0._R_P ; self%lt2(lv)%v = 0._R_P
      n = count(self%leaf_lev(1:self%nleaf_loc) == lv) * self%ncb
      allocate(self%mlev(lv)%v(n), self%mleaf(lv)%v(n))
      c = 0
      do b=1, self%nleaf_loc
         if (self%leaf_lev(b) /= lv) cycle
         q = self%leaf_lev_loc(b)
         do i=1, self%ncb
            c = c + 1
            self%mlev(lv)%v(c) = (q - 1) * self%ncb + i
            self%mleaf(lv)%v(c) = (b - 1) * self%ncb + i
         enddo
      enddo
   enddo
   n = self%nloc_lev(self%lbot)
   allocate(self%root_coords(3,max(n,1)), self%root_buf(max(n * self%ncb_of(self%lbot),1)), &
            self%root_out(max(n * self%ncb_of(self%lbot),1)))
   self%root_coords = 0 ; self%root_buf = 0._R_P ; self%root_out = 0._R_P
   do q=1, n
      self%root_coords(:,q) = self%node_c(1:3,self%lev_node(q,self%lbot))
   enddo
   ns = max(size(self%a%sidx), size(self%g%sidx))
   nh = max(self%a%nhalo, self%g%nhalo)
   do lv=self%lbot+1, self%lmax
      ns = max(ns, size(self%al(lv)%sidx), size(self%el(lv)%sidx), size(self%rl(lv)%sidx), size(self%pl(lv)%sidx), &
               size(self%etl(lv)%sidx))
      nh = max(nh, self%al(lv)%nhalo, self%el(lv)%nhalo, self%rl(lv)%nhalo, self%pl(lv)%nhalo, self%etl(lv)%nhalo)
   enddo
   allocate(self%xsend(max(ns,1)), self%xhalo(max(nh,1)))
   self%xsend = 0._R_P ; self%xhalo = 0._R_P
   endsubroutine make_work

   pure function lin(self, ijk) result(l)
   !< Linear index of cell (i,j,k) in a block, i fastest.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: ijk(3) !< Cell (1-based).
   integer(I4P)                         :: l      !< Index.

   l = ijk(1) + (ijk(2) - 1) * self%nijk(1) + (ijk(3) - 1) * self%nijk(1) * self%nijk(2)
   endfunction lin

   pure function lin_of(self, lv, ijk) result(l)
   !< Linear index of a cell (1-based) in a block at level lv (`nijk_of(lv)` cells).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv     !< Level.
   integer(I4P),             intent(in) :: ijk(3) !< Cell (1-based).
   integer(I4P)                         :: l      !< Index.
   integer(I4P)                         :: nl(3)  !< Cells per block.

   nl = self%nijk_of(lv)
   l = ijk(1) + (ijk(2) - 1) * nl(1) + (ijk(3) - 1) * nl(1) * nl(2)
   endfunction lin_of

   pure function ncb_of(self, lv) result(n)
   !< Cells per block at level lv.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv   !< Level.
   integer(I4P)                         :: n    !< Cells.

   n = product(self%nijk_of(lv))
   endfunction ncb_of

   pure function ncell(self, lv, d) result(n)
   !< Cells along direction d over the domain at level lv: the blocks at max(lv, lmin) times their cells.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv   !< Level.
   integer(I4P),             intent(in) :: d    !< Direction.
   integer(I4P)                         :: n    !< Cells.
   integer(I4P)                         :: nl(3) !< Cells per block.

   nl = self%nijk_of(lv)
   n = self%nblk0(d) * 2**max(lv, self%lmin) * nl(d)
   endfunction ncell

   pure function nijk_of(self, lv) result(nl)
   !< Cells per block per direction at level lv: nijk at lmin and above, halved per level below it along the active
   !< directions (the levels inside the blocks, #84 E5c).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv    !< Level.
   integer(I4P)                         :: nl(3) !< Cells per block.

   nl = self%nijk
   if (lv < self%lmin) nl = merge(self%nijk, self%nijk / 2**(self%lmin - lv), self%null)
   endfunction nijk_of

   pure subroutine locate(self, space, key, owner, lidx)
   !< Owner rank and local index of a cell key in a space (the composite leaves, or a level); owner -1 when the key
   !< names no leaf or node of the space (an internal invariant: the caller returns ELISHA_ERR_INTERNAL).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in)  :: space !< Space.
   integer(I8P),             intent(in)  :: key   !< Cell key.
   integer(I4P),             intent(out) :: owner !< Owner rank (-1: none).
   integer(I4P),             intent(out) :: lidx  !< Local index on the owner.
   integer(I4P)                          :: lv, cc(3), id, loc !< Decoded key, ids.
   integer(I4P)                          :: nl(3)              !< Cells per block at the level.

   owner = -1 ; lidx = 0
   call elisha_kdecode(key, lv, cc)
   if (space == SPACE_COMP) then
      id = self%leaf_hash%get(elisha_bkey([cc / self%nijk, lv]))
      if (id == 0) return ! a composite key names no leaf
      owner = self%leaf_owner(id)
      loc = self%leaf_loc(id)
   elseif (lv < self%lmin) then ! a level inside the blocks (#84 E5c): the lmin block holding the cell
      if (lv /= space) return ! a level key names another level
      nl = self%nijk_of(lv)
      id = self%node_hash%get(elisha_bkey([cc / nl, self%lmin]))
      if (id == 0) return ! a level key names no node
      owner = self%node_owner(id)
      lidx = (self%node_loc(id) - 1) * self%ncb_of(lv) + self%lin_of(lv, cc - (cc / nl) * nl + 1)
      return
   else
      if (lv /= space) return ! a level key names another level
      id = self%node_hash%get(elisha_bkey([cc / self%nijk, lv]))
      if (id == 0) return ! a level key names no node
      owner = self%node_owner(id)
      loc = self%node_loc(id)
   endif
   lidx = (loc - 1) * self%ncb + self%lin(cc - (cc / self%nijk) * self%nijk + 1)
   endsubroutine locate

   pure function parent3(self, cc) result(pc)
   !< Parent cell coordinates (one level coarser) along the active directions.
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: cc(3) !< Cell coordinates.
   integer(I4P)                         :: pc(3) !< Parent.

   pc = merge(cc, cc / 2, self%null)
   endfunction parent3

   subroutine smooth(self, lv, e, rhs, order, stat)
   !< Red-black Gauss-Seidel sweeps on level lv: A_l e = rhs (the ghost term already in rhs). The exchange buffers are
   !< the solver's (`xsend`, `xhalo`, audit C1).
   class(elisha_hierarchical_solver_object), intent(inout) :: self !< The solver.
   integer(I4P),             intent(in)    :: lv      !< Level.
   real(R_P),                intent(inout) :: e(1:)   !< Correction.
   real(R_P),                intent(in)    :: rhs(1:) !< Right-hand side.
   integer(I4P),             intent(in)    :: order   !< 0: colour 0 first, 1: colour 1 first.
   integer(I4P),             intent(out)   :: stat    !< Status.
   integer(I4P)                            :: sw, cc_, colr, i, k, p !< Counters.
   integer(I4P)                            :: q, ii, jj, kk !< Block, cell (stencil rows).
   real(R_P)                               :: acc     !< Off-diagonal sum.

   stat = ELISHA_OK
   associate(m=>self%al(lv), sbuf=>self%xsend, halo=>self%xhalo)
   do sw=1, self%sweeps
      do cc_=0, 1
         colr = mod(cc_ + order, 2)
         do i=1, size(m%sidx)
            sbuf(i) = e(m%sidx(i))
         enddo
         call m%plan%exchange_r(sbuf, halo, stat)
         if (stat /= ELISHA_OK) return
         if (.not.m%stencil) then ! ADAM's loop, unchanged
            do i=1, m%nrow
               if (self%colour(lv)%v(i) /= colr) cycle
               acc = 0._R_P
               do k=m%rowptr(i), m%rowptr(i+1)-1
                  if (m%col(k) == i) cycle
                  if (m%col(k) <= m%nloc) then
                     acc = acc + m%val(k) * e(m%col(k))
                  else
                     acc = acc + m%val(k) * halo(m%col(k) - m%nloc)
                  endif
               enddo
               e(i) = (rhs(i) - acc) / self%diag(lv)%v(i)
            enddo
         else ! the block-interior rows as the stencil, the neighbours in their CSR entries' order (M7, audit C2)
            do q=1, m%snblk
               do kk=1, m%snl(3)
                  do jj=1, m%snl(2)
                     do ii=1, m%snl(1)
                        i = (q - 1) * m%snl(1) * m%snl(2) * m%snl(3) + ii + (jj - 1) * m%snl(1) + &
                            (kk - 1) * m%snl(1) * m%snl(2)
                        if (self%colour(lv)%v(i) /= colr) cycle
                        if (ii < m%slo(1) .or. ii > m%shi(1) .or. jj < m%slo(2) .or. jj > m%shi(2) .or. &
                            kk < m%slo(3) .or. kk > m%shi(3)) then
                           acc = csr_off(i)
                        else
                           acc = 0._R_P
                           do p=1, m%sno
                              acc = acc + m%sov(p) * e(i + m%soo(p))
                           enddo
                        endif
                        e(i) = (rhs(i) - acc) / self%diag(lv)%v(i)
                     enddo
                  enddo
               enddo
            enddo
         endif
      enddo
   enddo
   endassociate
   contains
      function csr_off(r) result(a)
      !< The off-diagonal sum of row r of A_l e from its CSR entries.
      integer(I4P), intent(in) :: r  !< Row.
      real(R_P)                :: a  !< Its value.
      integer(I4P)             :: k_ !< Counter.

      a = 0._R_P
      associate(m_=>self%al(lv))
      do k_=m_%rowptr(r), m_%rowptr(r+1)-1
         if (m_%col(k_) == r) cycle
         if (m_%col(k_) <= m_%nloc) then
            a = a + m_%val(k_) * e(m_%col(k_))
         else
            a = a + m_%val(k_) * self%xhalo(m_%col(k_) - m_%nloc)
         endif
      enddo
      endassociate
      endfunction csr_off
   endsubroutine smooth

   subroutine vcycle(self, r_leaf, z_leaf, stat, errmsg)
   !< The preconditioner: z = M r, one V-cycle over the levels (see the module), down to the root (lbot), solved
   !< exactly by the spectral solver on every rank (#84 E5c: below lmin the levels coarsen inside the blocks). On the
   !< work vectors, through the hooks (#84 E6a); the root goes through the host.
   class(elisha_hierarchical_solver_object), intent(inout) :: self !< The solver.
   real(R_P),                intent(in)    :: r_leaf(1:) !< Leaf residual (V-weighted).
   real(R_P),                intent(inout) :: z_leaf(1:) !< Leaf correction.
   integer(I4P),             intent(out)   :: stat       !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
   integer(I4P)              :: lv, n, nc                 !< Counters.
   real(R_P)                 :: v0                        !< Root cell volume.

   do lv=self%lbot, self%lmax
      call self%vec_fill(self%lr(lv)%v, 0._R_P, size(self%lr(lv)%v))
      call self%vec_fill(self%le(lv)%v, 0._R_P, size(self%le(lv)%v))
   enddo
   ! the leaf residual into the level spaces
   do lv=self%lbot, self%lmax
      call self%vec_permute(r_leaf, self%mleaf(lv)%v, self%lr(lv)%v, self%mlev(lv)%v, size(self%mlev(lv)%v))
   enddo
   ! down
   do lv=self%lmax, self%lbot+1, -1
      n = self%nloc_lev(lv) * self%ncb_of(lv)
      nc = self%nloc_lev(lv-1) * self%ncb_of(lv-1)
      call self%smooth(lv, self%le(lv)%v, self%lr(lv)%v, 0_I4P, stat) ; if (stat /= ELISHA_OK) return
      call self%apply(self%al(lv), self%le(lv)%v, self%lt1(lv)%v, stat) ; if (stat /= ELISHA_OK) return
      call self%vec_lincomb(self%lr(lv)%v, -1._R_P, self%lt1(lv)%v, self%lt2(lv)%v, n)
      call self%apply(self%rl(lv), self%lt2(lv)%v, self%lt1(lv-1)%v, stat) ; if (stat /= ELISHA_OK) return
      call self%vec_axpy(1._R_P, self%lt1(lv-1)%v, self%lr(lv-1)%v, nc)
      if (self%vcycle_form == ELISHA_VCYCLE_SYMMETRIC .and. lv > self%lmin) then ! the adjoint of the way up's E_l
         call self%apply(self%etl(lv), self%le(lv)%v, self%lt1(lv-1)%v, stat) ; if (stat /= ELISHA_OK) return
         call self%vec_axpy(-1._R_P, self%lt1(lv-1)%v, self%lr(lv-1)%v, nc)
      endif
   enddo
   ! the root: exact, by the spectral solver on every rank (L e = r / V, the mean removed), through the host
   n = self%nloc_lev(self%lbot)
   nc = n * self%ncb_of(self%lbot)
   v0 = self%cell_volume(self%lbot)
   call self%root_solve(n, nc, v0, stat, errmsg) ; if (stat /= ELISHA_OK) return
   ! up
   do lv=self%lbot+1, self%lmax
      n = self%nloc_lev(lv) * self%ncb_of(lv)
      call self%apply(self%pl(lv), self%le(lv-1)%v, self%lt1(lv)%v, stat) ; if (stat /= ELISHA_OK) return
      call self%vec_axpy(1._R_P, self%lt1(lv)%v, self%le(lv)%v, n)
      if (lv > self%lmin) then ! a face to a cell outside the level reads the coarser correction
         call self%apply(self%el(lv), self%le(lv-1)%v, self%lt2(lv)%v, stat) ; if (stat /= ELISHA_OK) return
         call self%vec_lincomb(self%lr(lv)%v, -1._R_P, self%lt2(lv)%v, self%lt1(lv)%v, n)
      else ! at lmin and below every cell is in the level
         call self%vec_copy(self%lr(lv)%v, self%lt1(lv)%v, n)
      endif
      call self%smooth(lv, self%le(lv)%v, self%lt1(lv)%v, 1_I4P, stat) ; if (stat /= ELISHA_OK) return
   enddo
   ! the leaf correction
   do lv=self%lbot, self%lmax
      call self%vec_permute(self%le(lv)%v, self%mlev(lv)%v, z_leaf, self%mleaf(lv)%v, size(self%mlev(lv)%v))
   enddo
   endsubroutine vcycle

   pure function outside(self, lv, cc) result(beyond)
   !< A cell position at level lv beyond a domain face along a bounded active direction (never along a periodic one).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv     !< Level.
   integer(I4P),             intent(in) :: cc(3)  !< Cell coordinates.
   logical                              :: beyond !< Result.
   integer(I4P)                         :: p, d   !< Counters.

   beyond = .false.
   do p=1, self%nd
      d = self%act(p)
      if (self%periodic(d)) cycle
      if (cc(d) < 0 .or. cc(d) >= self%ncell(lv, d)) beyond = .true.
   enddo
   endfunction outside

   pure function wrap(self, lv, cc) result(cw)
   !< Periodic wrap of cell coordinates at level lv along the periodic active directions (bounded ones unchanged).
   class(elisha_hierarchical_solver_object), intent(in) :: self !< The solver.
   integer(I4P),             intent(in) :: lv    !< Level.
   integer(I4P),             intent(in) :: cc(3) !< Cell coordinates.
   integer(I4P)                         :: cw(3) !< Wrapped.
   integer(I4P)                         :: p, d  !< Counters.

   cw = cc
   do p=1, self%nd
      d = self%act(p)
      if (.not.self%periodic(d)) cycle
      cw(d) = modulo(cc(d), self%ncell(lv, d))
   enddo
   endfunction wrap

   ! module helpers
   function gdot(comm, x, y, n, stat) result(s)
   !< sum_i x_i y_i over the local n entries and the ranks (sum_real8: ADAM's MPI_REAL8 reduction).
   type(elisha_comm_object), intent(in)  :: comm  !< Communication context.
   real(R_P),                intent(in)  :: x(1:) !< First vector.
   real(R_P),                intent(in)  :: y(1:) !< Second vector.
   integer(I4P),             intent(in)  :: n     !< Local entries.
   integer(I4P),             intent(out) :: stat  !< Status.
   real(R_P)                             :: s     !< Result.
   real(R_P)                             :: x1(1) !< s, reduced in place.
   integer(I4P)                          :: i     !< Counter.

   s = 0._R_P
   do i=1, n
      s = s + x(i) * y(i)
   enddo
   x1 = s
   call comm%sum_real8(x1, stat)
   s = x1(1)
   endfunction gdot

   subroutine trip_add(t, row, key, val)
   !< Append a triplet.
   type(triplets), intent(inout) :: t     !< Buffer.
   integer(I4P),   intent(in)    :: row   !< Row.
   integer(I8P),   intent(in)    :: key   !< Column key.
   real(R_P),      intent(in)    :: val   !< Value.
   integer(I4P),   allocatable   :: r2(:) !< Grown rows.
   integer(I8P),   allocatable   :: k2(:) !< Grown keys.
   real(R_P),      allocatable   :: v2(:) !< Grown values.

   if (.not.allocated(t%r)) then
      allocate(t%r(1024), t%k(1024), t%v(1024))
   elseif (t%n == size(t%r)) then
      allocate(r2(2*t%n), k2(2*t%n), v2(2*t%n))
      r2(1:t%n) = t%r ; k2(1:t%n) = t%k ; v2(1:t%n) = t%v
      call move_alloc(r2, t%r) ; call move_alloc(k2, t%k) ; call move_alloc(v2, t%v)
   endif
   t%n = t%n + 1
   t%r(t%n) = row
   t%k(t%n) = key
   t%v(t%n) = val
   endsubroutine trip_add

   subroutine agree_status(comm, stat, errmsg, where)
   !< Every rank takes the largest status over the ranks (`max_i4`; the codes are positive): a rank that failed keeps
   !< its message, the others get the code with "on another rank". Collective.
   type(elisha_comm_object),  intent(in)              :: comm   !< Communication context.
   integer(I4P),              intent(inout)           :: stat   !< Status.
   character(:), allocatable, intent(inout), optional :: errmsg !< Message.
   character(*),              intent(in)              :: where  !< The procedure, for the message.
   integer(I4P)                                       :: c(1)   !< Agreed code.
   integer(I4P)                                       :: st     !< Status of the reduction.

   c = stat
   call comm%max_i4(c, st)
   if (st /= ELISHA_OK) then
      call elisha_set_error(stat, errmsg, st, where//': a collective failed')
   elseif (c(1) /= stat) then
      call elisha_set_error(stat, errmsg, c(1), where//': on another rank')
   endif
   endsubroutine agree_status
endmodule elisha_hierarchical_solver
