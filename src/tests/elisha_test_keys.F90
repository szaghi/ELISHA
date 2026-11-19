!< ELISHA test: the cell keys and the hash table, by their meaning and bitwise against ADAM's.
program elisha_test_keys
!< ELISHA test: the cell keys and the hash table, by their meaning and bitwise against ADAM's.
!<
!< - keys: `elisha_bkey`/`elisha_ckey` equal ADAM's `bkey`/`ckey` on a grid of coordinates up to 2^19 - 1 and levels
!<   -16..47 (cell keys, `ELISHA_KEY_LEVEL_MAX`) and 0..63 (block keys), every key non-negative (the int64 sign bit
!<   bounds the level field to 0..63, not the 7 bits ADAM's comment states); `elisha_kdecode` inverts `elisha_ckey`; distinct cells give
!<   distinct keys (a full sweep of a small box over several levels);
!< - table: 20000 pseudo-random keys and a run of keys congruent modulo the capacity (worst-case probing) put and got
!<   back; absent keys give 0; an overwrite replaces; the capacity is the first prime above 2 n, as ADAM's.
use elisha_keys, only : ELISHA_KEY_COORD_LIMIT, ELISHA_KEY_LEVEL_OFFSET, ELISHA_KEY_LEVEL_MAX, elisha_bkey, &
                        elisha_ckey, elisha_kdecode, elisha_hash_table
use adam_flail_keys_reference, only : KEY_LEVEL_OFFSET, hash_table, bkey, ckey, kdecode, hash_init, hash_put, hash_get
use penf, only : I4P, I8P
implicit none
integer(I4P), parameter :: NK=20000_I4P            !< Random keys.
type(elisha_hash_table) :: h                      !< ELISHA's table.
type(hash_table)        :: ha                     !< ADAM's table.
integer(I4P)            :: failures               !< Failed checks.
integer(I4P)            :: lv, i, j, k, l2, c2(3), n, v !< Counters, decoded values.
integer(I4P)            :: cs(6)                  !< Coordinate samples.
integer(I8P)            :: key, seed              !< Key, generator state.
integer(I8P), allocatable :: keys(:), seen(:)     !< Keys.

failures = 0
if (ELISHA_KEY_LEVEL_OFFSET /= KEY_LEVEL_OFFSET) call fail('level offset')
cs = [0, 1, 2, 12345, ELISHA_KEY_COORD_LIMIT - 2, ELISHA_KEY_COORD_LIMIT - 1]
do lv=-ELISHA_KEY_LEVEL_OFFSET, ELISHA_KEY_LEVEL_MAX
   do k=1, 6 ; do j=1, 6 ; do i=1, 6
      key = elisha_ckey(lv, [cs(i), cs(j), cs(k)])
      if (key /= ckey(lv, [cs(i), cs(j), cs(k)])) call fail('cell key not ADAM''s')
      if (key < 0_I8P) call fail('negative key')
      call elisha_kdecode(key, l2, c2)
      if (l2 /= lv .or. any(c2 /= [cs(i), cs(j), cs(k)])) call fail('decode does not invert')
   enddo ; enddo ; enddo
enddo
do lv=0, 63
   if (elisha_bkey([cs(5), cs(2), cs(6), lv]) /= bkey([cs(5), cs(2), cs(6), lv])) call fail('block key not ADAM''s')
enddo
allocate(seen(0))
do lv=-2, 2
   do k=0, 7 ; do j=0, 7 ; do i=0, 7
      seen = [seen, elisha_ckey(lv, [i, j, k])]
   enddo ; enddo ; enddo
enddo
do i=2, size(seen)
   if (any(seen(1:i-1) == seen(i))) then
      call fail('two cells, one key')
      exit
   endif
enddo
! the table
allocate(keys(NK))
seed = 88172645463325252_I8P
do i=1, NK
   seed = ieor(seed, ishft(seed, 13)) ; seed = ieor(seed, ishft(seed, -7)) ; seed = ieor(seed, ishft(seed, 17))
   keys(i) = iand(seed, huge(1_I8P))
enddo
call h%init(NK) ; call hash_init(ha, NK)
if (h%cap /= ha%cap) call fail('capacity not ADAM''s')
if (.not.is_prime(h%cap) .or. h%cap <= 2 * NK) call fail('capacity not a prime above 2 n')
do i=1, NK
   call h%put(keys(i), i) ; call hash_put(ha, keys(i), i)
enddo
do i=1, NK
   v = h%get(keys(i))
   if (v /= i .and. any(keys(1:i-1) == keys(i))) cycle ! a repeated random key keeps its last value
   if (v /= i) call fail('a key lost')
   if (v /= hash_get(ha, keys(i))) call fail('get not ADAM''s')
enddo
if (any(h%keys /= ha%keys) .or. any(h%vals /= ha%vals)) call fail('table layout not ADAM''s')
n = 0
do i=1, 1000
   if (any(keys == int(i, I8P))) cycle
   if (h%get(int(i, I8P)) /= 0) n = n + 1
enddo
if (n > 0) call fail('absent keys found')
call h%put(keys(7), -3) ; if (h%get(keys(7)) /= -3) call fail('overwrite')
call h%init(50) ! keys congruent modulo the capacity: one probe chain
do i=1, 50
   call h%put(int(i, I8P) * h%cap, i)
enddo
do i=1, 50
   if (h%get(int(i, I8P) * h%cap) /= i) call fail('collision chain')
enddo
if (h%get(51_I8P * h%cap) /= 0) call fail('absent key on a full chain')
call h%free ; if (h%cap /= 0 .or. allocated(h%keys)) call fail('free')
if (failures > 0) error stop 1
print '(A)', 'elisha_test_keys: ok'
contains
   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   if (failures <= 10) print '(A)', '  FAIL '//what
   endsubroutine fail

   pure function is_prime(c) result(p)
   !< Primality by trial division.
   integer(I4P), intent(in) :: c !< Candidate.
   logical                  :: p !< Prime.
   integer(I4P)             :: f !< Factor.

   p = c > 1
   do f=2, int(sqrt(real(c))) + 1
      if (f < c .and. mod(c, f) == 0) p = .false.
   enddo
   endfunction is_prime
endprogram elisha_test_keys
