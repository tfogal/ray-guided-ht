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

__device__ static unsigned next = 0x12345678u;
__device__ static unsigned
devrand()
{
	next = next * 1103515245 + 12345;
	return ((unsigned)(next/2147483648) % 1073741824);
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


/** @param ht the hash table
 * @param ??? the dimensions of the hash table, in shared mem
 * @param list of bricks to access.  this is 4-components! (x,y,z, LOD) */
__global__ void
ht_inserts(unsigned* ht, const size_t htlen, const uint32_t* bricks,
           const size_t nbricks)
{
	/* shared memory for writes which should get added to 'ht'. */
	__shared__ unsigned pending[16];
	__shared__ unsigned pidx;

	/* __shared__ vars can't have initializers; do it manually. */
	for(size_t i=0; i < 16; i++) { pending[i] = 0; }
	pidx = 0;
	__syncthreads();

	for(size_t i=0; i < MAX_BRICK_REQUESTS; ++i) {
		const unsigned bid = devrand() % nbricks;
		unsigned serialized = serialize(&bricks[bid*4], brickdims);

		/* Is it already in the table?  then move on. */
		if(find_entry(ht, htlen, serialized)) { continue; }

		/* Otherwise, add it to our list of pending writes into the
		 * table.  But, that might cause it to overflow, which means
		 * we'd have to flush it. */
		if(pidx >= 16) {
			flush(ht, htlen, pending, 16);
			atomicCAS(&pidx, 16U, 0U);
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
		const unsigned bid = devrand() % nbricks;
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
			fprintf(stderr, "Error scanning request %zu: %d\n",
			        req, errno);
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

	fprintf(stderr, "%zu-element hash table.\n", N_ht);
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
	fprintf(stderr, "dev alloc ht okay\n");

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

	fprintf(stderr, "dev alloc bricks okay\n");

	err = cudaMemcpy(bricks_dev, bricks_host, brickbytes,
	                 cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "cuda copy error (bricks) host -> dev: %s\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	fprintf(stderr, "bricks cpy okay\n");

	dim3 blocks(60, 33);
#if 1
	ht_inserts<<<blocks, 32>>>(htable_dev, N_ht, bricks_dev, nrequests);
#else
	ht_inserts_simple<<<blocks, 32>>>(htable_dev, N_ht, bricks_dev,
	                                  nrequests);
#endif

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
