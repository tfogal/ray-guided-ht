#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "ba-file.h"
#include "compiler.h"
#include "opt.h"
#include "requests.h"
#include "table.h"

PURE __device__ static uint32_t
serialize(const uint32_t bidx[4], const unsigned bdims[4])
{
	return 1 + bidx[0] + bidx[1]*bdims[0] + bidx[2]*bdims[0]*bdims[1] +
	       bidx[3]*bdims[0]*bdims[1]*bdims[2];
}

/* 16384^3 volume / 32^3 voxel bricks == 512^3 bricks.  So an
 * axis-aligned ray (i.e. a thread) couldn't request more than 512
 * bricks. */
#ifndef MAX_BRICK_REQUESTS
#	define MAX_BRICK_REQUESTS 512U
#endif

__constant__ unsigned brickdims[4] = {0};

/* try to find the given value in the table.  this may not occur at the hashed
 * position, of course, since collisions may occur.  it would be at subsequent
 * elements, then. */
__device__ static bool
find_entry(unsigned* ht, const size_t htlen, unsigned value)
{
#ifndef ELEMS_TO_SEARCH
#	define ELEMS_TO_SEARCH 10
#endif
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
#ifndef MAX_ITERS
#	define MAX_ITERS 10
#endif
	for(size_t i=0; i < n; ++i) {
		size_t iter = 0;
		do {
			const unsigned hpos = (pending[i] + iter) % htlen;
			uint32_t value = atomicCAS(&ht[hpos], 0U, pending[i]);
			if(value == 0 || value == pending[i]) { break; }
		} while(++iter < MAX_ITERS);
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
			atomicInc(&pidx, pidx);
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

/** loads brick requests from the given file.  This could be either our custom
 * '.req' file, or a '.ba' file.  The return value is an array with all the
 * brick requests, with each brick request being a 4D brick index.
 * @param[in] filename file to load from
 * @param[out] n_requests the number of requests that we can load. */
static unsigned*
load_brick_requests(const char* filename, size_t* n_requests)
{
	/* guess file type based on extension. */
	const char* dot = strrchr(filename, '.');
	if(dot == NULL) {
		errno = EINVAL;
		*n_requests = 0;
		return NULL;
	}

	if(*(dot+1) == 'r') {
		return requests_from(filename, n_requests);
	} else if(*(dot+1) == 'b') {
		return requests_ba(filename, n_requests);
	}
	*n_requests = 0;
	errno = EINVAL;
	return NULL;
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
	unsigned* bricks_host = load_brick_requests(requestfile(), &nrequests);
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
	const dim3 blocks(bdims[0]/bf, bdims[1]/bf, bdims[2]/bf);
	while(nrequests > 0) {
		if(verbose()) { printf("launching kernel...\n"); }
		if(naive()) {
			ht_inserts_simple<<<blocks, 128>>>(htable_dev, N_ht,
			                                   bricks_dev,
							   nrequests);
		} else {
			ht_inserts<<<blocks, 128>>>(htable_dev, N_ht,
			                            bricks_dev, nrequests);
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
		printf("Removing entries from HT in %zu-elem request pool.\n",
		       nrequests);
		nrequests = remove_entries(htable_host, N_ht, bricks_host, nrequests,
					   main_brickdims);
		printf("%zu requests left.\n", nrequests);

		if(duplicates(htable_host, N_ht)) {
			fprintf(stderr, "Something broke; duplicates!\n");
			exit(EXIT_FAILURE);
		}

		memset(htable_host, 0, N_ht*sizeof(unsigned));
		err = cudaMemcpy(htable_dev, htable_host,
		                 N_ht*sizeof(unsigned),
		                 cudaMemcpyHostToDevice);
		if(err != cudaSuccess) {
			fprintf(stderr, "error fixing HT..\n");
			exit(EXIT_FAILURE);
		}

		/* copy modified brick requests back to device buffer. */
		err = cudaMemcpy(bricks_dev, bricks_host, brickbytes,
				 cudaMemcpyHostToDevice);
		if(err != cudaSuccess) {
			fprintf(stderr, "cuda copy error (bricks) host -> "
			        "dev: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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
