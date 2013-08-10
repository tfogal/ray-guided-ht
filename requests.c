#include <errno.h>
#include <stdio.h>
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
	if(fscanf(fp, "%u\n", nreqs) != 1) {
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
			fprintf(stderr, "Error scanning request %u(%d): %d\n",
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
