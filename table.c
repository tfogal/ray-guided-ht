#include <stdio.h>
#include "compiler.h"
#include "opt.h"
#include "table.h"

/* removes index 'idx' from the given brick list. */
PURE static void
remove_entry(size_t idx, unsigned* bricks, const size_t n_bricks)
{
	for(size_t i=idx; i < n_bricks-1; ++i) {
		bricks[i*4+0] = bricks[(i+1)*4+0];
		bricks[i*4+1] = bricks[(i+1)*4+1];
		bricks[i*4+2] = bricks[(i+1)*4+2];
		bricks[i*4+3] = bricks[(i+1)*4+3];
	}
}

/** returns the index into 'bricks' where 'brickID' lies.  returns n_bricks if
 * the entry was not found. */
PURE static size_t
idx_for_brick(const unsigned brickID[4], const unsigned* bricks, const size_t n_bricks)
{
	for(size_t i=0; i < n_bricks; ++i) {
		if(brickID[0] == bricks[i*4+0] &&
		   brickID[1] == bricks[i*4+1] &&
		   brickID[2] == bricks[i*4+2] &&
		   brickID[3] == bricks[i*4+3]) {
			return i;
		}
	}
	return n_bricks;
}

/** @param idx1 the 1D index of the brick (what got stored in the HT)
 * @param[out] bid output.  the corresponding 4D index from 'idx1'.
 * @param[in] bdims the number of bricks in each dimension */
void
to4d(const unsigned idx1, unsigned bid[4], const unsigned bdims[4])
{
	bid[0] = idx1 % bdims[0];
	bid[1] = (idx1 / bdims[0]) % bdims[1];
	bid[2] = (idx1 / (bdims[0]*bdims[1])) % bdims[2];
	/* One really doesn't need the mod operation in the last elem.. */
	bid[3] = (idx1 / (bdims[0]*bdims[1]*bdims[2])) % bdims[3];
}

/** removes a set of HT entries from the given bricktable.
 * @param entries the entries.  note these are 1D (collapsed) indices.
 * @param n_entries number of hash table entries
 * @param bricks the set of bricks, the brick table.
 * @param n_bricks number of bricks.  each brick is 4 elems!
 * @param bdims the number of bricks in each dimension
 * @returns the modified/new number of bricks in the table. */
size_t
remove_entries(const unsigned* entries, const size_t n_entries,
               unsigned* bricks, size_t n_bricks,
               const unsigned bdims[4])
{
	for(size_t i=0; i < n_entries; ++i) {
		if(entries[i] > 0) {
			size_t count = 0;
                        unsigned brickID[4];
                        to4d(entries[i], brickID, bdims);
			do {
				size_t rem = idx_for_brick(brickID, bricks,
				                           n_bricks);
				remove_entry(rem, bricks, n_bricks);
				n_bricks--;
				count++;
			} while(idx_for_brick(brickID, bricks, n_bricks) !=
			        n_bricks);
			if(verbose()) {
				printf("Removed %zu bricks for %u\n", count,
				       entries[i]);
			}
		}
	}
	return n_bricks;
}

PURE size_t
nonzeroes(const unsigned* ht, const size_t n_entries)
{
	size_t count = 0;
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] != 0) { ++count; }
	}
	return count;
}

PURE static size_t
count(unsigned value, const unsigned* ht, const size_t n_entries)
{
	size_t n=0;
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] == value) { ++n; }
	}
	return n;
}

PURE bool
duplicates(const unsigned* ht, const size_t n_entries)
{
	for(size_t i=0; i < n_entries; ++i) {
		if(count(ht[i], ht, n_entries) > 1) {
			if(verbose()) {
				fprintf(stderr, "%u appears %zu times!\n",
				        ht[i], count(ht[i], ht, n_entries));
				return true;
			}
		}
	}
	return false;
}
