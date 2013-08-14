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
#	define ELEMS_TO_SEARCH 1
#endif
	for(size_t i=0; i < ELEMS_TO_SEARCH; ++i) {
		const unsigned idx = (value + i) % htlen;
		if(ht[idx] == value) { return true; }
	}
	return false;
}

#ifndef MAX_ITERS
#	define MAX_ITERS ELEMS_TO_SEARCH + 8
#endif

__device__ static void
flush_elem(unsigned* ht, const size_t htlen, unsigned elem)
{
	size_t iter = 0;
	do {
		const unsigned hpos = (elem + iter) % htlen;
		const unsigned value = atomicCAS(&ht[hpos], 0U, elem);
		if(value == 0 || value == elem) { break; }
	} while(++iter < MAX_ITERS);
}

#ifndef PENDING
#	define PENDING 1U
#endif
/* flushes all the entries from 'pending' to the hash table. */
__device__ static void
flush(unsigned* ht, const size_t htlen, const unsigned pending[PENDING],
      const size_t n)
{
	for(size_t i=0; i < n; ++i) {
		flush_elem(ht, htlen, pending[i]);
		/* We could atomicExch pending[i] back to 0 now.. but there's
		 * not really a point. */
		/* atomicExch(&pending[i], 0U); */
	}
}

PURE __device__ static unsigned
minu32(unsigned a, unsigned b)
{
	return a < b ? a : b;
}

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

	for(size_t i=0; i < MAX_BRICK_REQUESTS; ++i) {
		const unsigned bid = ((threadIdx.x + blockDim.x*blockDim.y) +
		                      i) % nbricks;
		unsigned serialized = serialize(&bricks[bid*4], brickdims);

		/* Is it already in the table?  then these aren't the bricks
		 * we're locking[sic] for.. We can go about our business. */
		if(find_entry(ht, htlen, serialized)) { continue; }

		/* Otherwise, add it to our list of pending writes into the
		 * table.  But, that might cause it to overflow, which means
		 * we'd have to flush it. */
		if(pidx >= PENDING) {
			flush(ht, htlen, pending, PENDING);
			atomicExch(&pidx, 0U);
		} else {
			atomicExch(&pending[pidx % PENDING], serialized);
			atomicAdd(&pidx, 1);
		}
	}
	flush(ht, htlen, pending, minu32(pidx, PENDING));
}

__global__ void
ht_inserts_nosync(unsigned* ht, const size_t htlen, const uint32_t* bricks,
                  const size_t nbricks)
{
	/* if NUMTHREADS is smaller than the number of threads which use the
	 * same __shared__ memory, all of this is broken. */
#define NUMLOCAL 16U
	unsigned pending[NUMLOCAL];
	unsigned pidx;

	/* __shared__ vars can't have initializers; do it manually. */
	for(size_t i=0; i < NUMLOCAL; ++i) { pending[i] = 0; }
	pidx = 0;

	for(size_t i=0; i < MAX_BRICK_REQUESTS; ++i) {
		const unsigned bid = ((threadIdx.x + blockDim.x*blockDim.y) +
		                      i) % nbricks;
		unsigned serialized = serialize(&bricks[bid*4], brickdims);

		/* Otherwise, add it to our list of pending writes into the
		 * table.  But, that might cause it to overflow, which means
		 * we'd have to flush it. */
		if(pidx == NUMLOCAL) {
			flush(ht, htlen, pending, NUMLOCAL);
			pidx = 0U;
		}
		pending[pidx] = serialized;
		pidx++;
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

		flush_elem(ht, htlen, serialized);
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

	/* each brick request is 16 bytes: 4 unsigned numbers (X,Y,Z,LOD) */
	const size_t brickbytes = nrequests * 4 * sizeof(uint32_t);

	uint32_t* bricks_dev;
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
	const dim3 blocks(bf, bf, bf);
	size_t iterations = 0;
	while(nrequests > 0) {
		++iterations;
		if(naive()) {
			ht_inserts_simple<<<blocks, 128>>>(htable_dev, N_ht,
			                                   bricks_dev,
							   nrequests);
		} else if(nosync()) {
			ht_inserts_nosync<<<blocks, 128>>>(htable_dev, N_ht,
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
		/* we added 1 to each entry when they went in the HT: subtract
		 * out that one. */
		subtract1(htable_host, N_ht);

		if(verbose()) {
			printf("Removing %zu HT entries from %zu-elem "
			       "request pool.\n", nonzeroes(htable_host, N_ht),
			       nrequests);
		}
		assert(nonzeroes(htable_host, N_ht) > 0);
		if(duplicates(htable_host, N_ht)) {
			fprintf(stderr, "Something broke; duplicates!\n");
			exit(EXIT_FAILURE);
		}

		nrequests = remove_entries(htable_host, N_ht, bricks_host, nrequests,
					   main_brickdims);
		if(!requests_verify(bricks_host, nrequests, main_brickdims,
		   NULL)) {
			fprintf(stderr, "Removing entries broke table.\n");
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

	printf("SUCCESS after %zu iterations.\n", iterations);

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
