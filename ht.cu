#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "opt.h"

#ifdef __GNUC__
#	define PURE __attribute__((pure))
#	define CONST __attribute__((const))
#else
#	define PURE /* no pure function support */
#	define CONST /* no const function support */
#endif

PURE __device__ static uint32_t
serialize(const uint32_t bidx[4], const unsigned bdims[4])
{
	return 1 + bidx[0] + bidx[1]*bdims[0] + bidx[2]*bdims[0]*bdims[1] +
	       bidx[3]*bdims[0]*bdims[1]*bdims[2];
}

/* 16384^3 volume / 32^3 voxel bricks == 512^3 bricks.  So an
 * axis-aligned ray (i.e. a thread) couldn't request more than 512
 * bricks. */
const size_t MAX_BRICK_REQUESTS = 512;

__constant__ unsigned brickdims[4] = {0};

/* try to find the given value in the table.  this may not occur at the hashed
 * position, of course, since collisions may occur.  it would be at subsequent
 * elements, then. */
__device__ static bool
find_entry(unsigned* ht, const size_t htlen, unsigned value)
{
#	define ELEMS_TO_SEARCH 4
	for(size_t i=0; i < ELEMS_TO_SEARCH; ++i) {
		const unsigned idx = (value + i) % htlen;
		if(ht[idx] == value) { return true; }
	}
	return false;
}

/* flushes all the entries from 'pending' to the hash table. */
__device__ static void
flush(unsigned* ht, const size_t htlen, unsigned pending[16], const size_t n)
{
	for(size_t i=0; i < n; ++i) {
		size_t iter = 0;
		do {
			const unsigned hpos = (pending[i] + iter) % htlen;
			uint32_t value = atomicCAS(&ht[hpos], 0U, pending[i]);
			if(value == 0 || value == pending[i]) { break; }
		} while(++iter < 10);
		/* We could atomicExch pending[i] back to 0 now.. but there's
		 * not really a point. */
		/* atomicExch(&pending[i], 0U); */
	}
}


#define PENDING 128U
/** @param ht the hash table
 * @param ??? the dimensions of the hash table, in shared mem
 * @param list of bricks to access.  this is 4-components! (x,y,z, LOD) */
__global__ void
ht_inserts(unsigned* ht, const size_t htlen, const uint32_t* bricks,
           const size_t nbricks)
{
	/* shared memory for writes which should get added to 'ht'. */
	__shared__ unsigned pending[PENDING];
	__shared__ unsigned pidx;

	/* __shared__ vars can't have initializers; do it manually. */
	for(size_t i=0; i < PENDING; i++) { pending[i] = 0; }
	pidx = 0;
	__syncthreads();

	for(size_t i=0; i < MAX_BRICK_REQUESTS; ++i) {
		const unsigned bid = ((threadIdx.x + blockDim.x*blockDim.y) +
		                      i) % nbricks;
		unsigned serialized = serialize(&bricks[bid*4], brickdims);

		/* Is it already in the table?  then move on. */
		if(find_entry(ht, htlen, serialized)) { continue; }

		/* Otherwise, add it to our list of pending writes into the
		 * table.  But, that might cause it to overflow, which means
		 * we'd have to flush it. */
		if(pidx >= PENDING) {
			flush(ht, htlen, pending, PENDING);
			atomicCAS(&pidx, PENDING, 0U);
		} else {
			atomicExch(&pending[pidx], serialized);
			atomicAdd(&pidx, 1);
		}
	}
	flush(ht, htlen, pending, pidx);
}

__global__ void
ht_inserts_simple(unsigned* ht, const size_t htlen, const uint32_t* bricks,
                  const size_t nbricks)
{
	for(size_t i=0; i < MAX_BRICK_REQUESTS; ++i) {
		const unsigned bid = ((threadIdx.x + blockDim.x*blockDim.y) +
		                      i) % nbricks;
		unsigned serialized = serialize(&bricks[bid*4], brickdims);

		unsigned rehash_count = 0;
		do {
			const unsigned hpos = (serialized + rehash_count) %
			                       htlen;
			unsigned val = atomicCAS(&ht[hpos], 0U, serialized);
			if(val == 0 || val == serialized) { break; }
		} while(++rehash_count < 10);
	}
}

/** reads requests from the given filename.
 * @returns the array of requests, or NULL on error.
 * @param[out] nreqs the number of requests in the array; note the array
 *             then has 4*requests elements, since each requests is 4
 *             entries. */
static uint32_t*
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

/* are the given requests valid?  they need to fall within brick indices.
 * @param requests the requests the examine
 * @param nreq number of requests; 'requests' is 4*nreq elems long.
 * @param bdims the brick dimensions.
 * @param[out] erridx if nonnull, the request which was in error. */
static bool
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

/** returns the index into 'bricks' where 'brickID' lies.  returns n_bricks if
 * the entry was not found. */
static size_t
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

size_t
nonzeroes(const unsigned* ht, const size_t n_entries)
{
	size_t count = 0;
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] != 0) { ++count; }
	}
	return count;
}

int
main(int argc, char* argv[])
{
	argparse(argc, argv);

	const size_t N_ht = htN();
	const unsigned main_brickdims[4] = { bricksX(), bricksY(), bricksZ(),
	                                     LODs() };

	cudaError_t cerr = cudaMemcpyToSymbol(brickdims, main_brickdims,
	                                      sizeof(unsigned)*4, 0,
	                                      cudaMemcpyHostToDevice);
	if(cerr != cudaSuccess) {
		fprintf(stderr, "could not copy brickdim data: %s\n",
		        cudaGetErrorString(cerr));
		exit(EXIT_FAILURE);
	}

	/* create our hash table, and a chunk of memory to read it back into
	 * when we're done.  We could also use pinned memory.. but...
	 * well, we should try that, too. */
	unsigned* htable_dev = NULL;
	cudaError_t err = cudaMalloc(&htable_dev,
	                             N_ht*sizeof(unsigned));
	if(err != cudaSuccess) {
		fprintf(stderr, "dev alloc of HT (size %zu) failed!: %s.\n",
		        N_ht, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	unsigned* htable_host = (unsigned*)calloc(sizeof(unsigned), N_ht);
	assert(htable_host); /* sometimes I <3 not being a real SW developer. */

	/* copy our (empty) hash table to the device: initialized to all 0s. */
	err = cudaMemcpy(htable_dev, htable_host,
	                 N_ht*sizeof(unsigned), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "cuda copy error host -> dev: %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	size_t nrequests;
	unsigned* bricks_host = requests_from(requestfile(), &nrequests);
	if(bricks_host == NULL) {
		fprintf(stderr, "Could not read requests from %s!\n",
		        requestfile());
		exit(EXIT_FAILURE);
	}
	size_t fault;
	if(!requests_verify(bricks_host, nrequests, main_brickdims, &fault)) {
		fprintf(stderr, "Brick request %zu is garbage.\n", fault);
		exit(EXIT_FAILURE);
	}
	const size_t brickbytes = nrequests * 4 * sizeof(uint32_t);

	uint32_t* bricks_dev;
	/* each brick request is 16 bytes: 4 unsigned numbers (X,Y,Z,LOD) */
	err = cudaMalloc(&bricks_dev, brickbytes);
	if(err != cudaSuccess) {
		fprintf(stderr, "cuda alloc error for bricks (dev): %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(bricks_dev, bricks_host, brickbytes,
	                 cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "cuda copy error (bricks) host -> dev: %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	const size_t bf = blockingfactor();
	const size_t bdims[] = { main_brickdims[0], main_brickdims[1],
	                         main_brickdims[2] };
	dim3 blocks(bdims[0]/bf, bdims[1]/bf, bdims[2]/bf);
	if(naive()) {
                ht_inserts_simple<<<blocks, 128>>>(htable_dev, N_ht, bricks_dev,
		                                   nrequests);
	} else {
                ht_inserts<<<blocks, 128>>>(htable_dev, N_ht, bricks_dev,
		                            nrequests);
	}

	if((err = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "Failed to launch kernel: %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	/* get the hash table back. */
	err = cudaMemcpy(htable_host, htable_dev,
	                 N_ht*sizeof(unsigned),
	                 cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "copy error (htable) dev -> host: %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("%zu nonzero entries in %zu-elem table.\n",
	       nonzeroes(htable_host, N_ht), N_ht);
	printf("Removing entries from the HT...\n");
	nrequests = remove_entries(htable_host, N_ht, bricks_host, nrequests,
	                           main_brickdims);
	printf("%zu requests left.\n", nrequests);

	printf("Test PASSED\n");

	if((err = cudaFree(htable_dev)) != cudaSuccess) {
		fprintf(stderr, "couldn't free device hash table! %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(htable_host);

	/* needed for benchmarks, though the reset afterwards probably makes it
	 * irrelevant. */
	cudaThreadSynchronize();
	if((err = cudaDeviceReset()) != cudaSuccess) {
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return 0;
}
