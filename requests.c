#include <errno.h>
#include <stdio.h>
#include "compiler.h"
#include "requests.h"

/** reads requests from the given filename.
 * @returns the array of requests, or NULL on error.
 * @param[out] nreqs the number of requests in the array; note the array
 *             then has 4*requests elements, since each requests is 4
 *             entries. */
uint32_t*
requests_from(const char* filename, size_t* nreqs)
{
	*nreqs = 0;
	FILE* fp = fopen(filename, "r");
	if(NULL == fp) {
		errno = EINVAL;
		return NULL;
	}
	if(fscanf(fp, "%zu\n", nreqs) != 1) {
		fclose(fp);
		errno = EPROTO;
		return NULL;
	}
	uint32_t* requests = (uint32_t*)malloc(sizeof(uint32_t)*4*(*nreqs));
	for(size_t req=0; req < *nreqs; ++req) {
		int scan = fscanf(fp, "%u %u %u %u\n", &requests[req*4+0],
		                  &requests[req*4+1], &requests[req*4+2],
		                  &requests[req*4+3]);
		if(scan != 4) {
			fprintf(stderr, "Error scanning request %zu(%d): %d\n",
			        req, scan, errno);
			fclose(fp);
			free(requests);
			*nreqs = 0;
			errno = EPROTO;
			return NULL;
		}
	}
	return requests;
}

PURE static uint32_t
serialize(const uint32_t bidx[4], const unsigned bdims[4])
{
	return bidx[0] + bidx[1]*bdims[0] + bidx[2]*bdims[0]*bdims[1] +
	       bidx[3]*bdims[0]*bdims[1]*bdims[2];
}

/* are the given requests valid?  they need to fall within brick indices.
 * @param requests the requests the examine
 * @param nreq number of requests; 'requests' is 4*nreq elems long.
 * @param bdims the brick dimensions.
 * @param[out] erridx if nonnull, the request which was in error. */
bool
requests_verify(const uint32_t* requests, const size_t nreq,
                const unsigned bdims[4], size_t* erridx)
{
	/* this actually isn't great, because we assume that the valid indices
	 * are 0 to bdims[0]*bdims[1]*bdims[2]*bdims[3].  In reality, the
	 * number of bricks decreases by half every time we drop to a coarser
	 * LOD, so there are far fewer bricks.
	 * This should at least catch the most egregious errors. */
	for(size_t r=0; r < nreq; ++r) {
		for(size_t dim=0; dim < 4; ++dim) {
			if(requests[r*4+dim] >= bdims[dim]) {
				if(erridx != NULL) { *erridx = r; }
				return false;
			}
		}
	}
	return true;
}

size_t
idx_of(const uint32_t* requests, const size_t nreq, const unsigned val,
       const unsigned bdims[4])
{
	size_t idx=nreq;
	for(size_t r=0; r < nreq; ++r) {
		const unsigned value = serialize(&requests[r*4], bdims);
		if(value == val) {
			printf("%u is at %zu\n", val, r);
			idx = r;
		}
	}
	return idx;
}

/* removes index 'idx' from the given brick list. */
static void
remove_entry(size_t idx, unsigned* bricks, const size_t n_bricks)
{
	for(size_t i=idx; i < n_bricks-1; ++i) {
		bricks[i*4+0] = bricks[(i+1)*4+0];
		bricks[i*4+1] = bricks[(i+1)*4+1];
		bricks[i*4+2] = bricks[(i+1)*4+2];
		bricks[i*4+3] = bricks[(i+1)*4+3];
	}
}

/** removes all entries from the request table which serialize to the given
 * value.
 * @param[in] serized the serialized value to be removed
 * @param[inout] bricks the table of bricks to modify
 * @param[in] n_bricks the number of bricks in the given table.
 * @return the new/updated number of bricks in the table */
size_t
remove_all(unsigned serized, unsigned* bricks, size_t n_bricks,
           const unsigned bdims[4])
{
	bool converged;
	do {
		converged = true;
		for(size_t i=0; i < n_bricks; ++i) {
			if(serialize(&bricks[i*4], bdims) == serized) {
				remove_entry(i, bricks, n_bricks);
				n_bricks--;
				converged = false;
				break;
			}
		}
	} while(!converged);
	return n_bricks;
}

void
write_requests(const char* file, const unsigned* bricks, size_t n_bricks)
{
	FILE* fp = fopen(file, "w+");
	if(!fp) {
		fprintf(stderr, "Error creating '%s'\n", file);
		return;
	}
	fprintf(fp, "%zu\n", n_bricks);
	for(size_t i=0; i < n_bricks; ++i) {
		fprintf(fp, "%u %u %u %u\n",
		        bricks[i*4+0], bricks[i*4+1],
		        bricks[i*4+2], bricks[i*4+3]);
	}
	fclose(fp);
}

/** creates duplicates of brick requests, increasing the list size.
 * @param[in] bricks the previous/existing brick requests
 * @param[inout] the number of bricks in the incoming list; modified to be the
 *               number of bricks in the outgoing list.
 * @returns the new list. */
MALLOC unsigned*
increase_requests(const unsigned* bricks, size_t* n_bricks)
{
	size_t new_bricks = (*n_bricks)*16;
	unsigned* bnew = malloc(sizeof(unsigned)*4*new_bricks);
	for(size_t i=0; i < new_bricks; ++i) {
		size_t idx = i % (*n_bricks);
		bnew[i*4+0] = bricks[idx*4+0];
		bnew[i*4+1] = bricks[idx*4+1];
		bnew[i*4+2] = bricks[idx*4+2];
		bnew[i*4+3] = bricks[idx*4+3];
	}
	*n_bricks = new_bricks;
	return bnew;
}
