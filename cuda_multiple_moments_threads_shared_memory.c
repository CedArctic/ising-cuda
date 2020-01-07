// Block size axis (BLOCK_SIZE^2 = number of threads per block)
#define BLOCK_SIZE 47
// Number of blocks on axis (GRID_SIZE^2 = number of blocks in grid)
//#define GRID_SIZE 11 // Number is now dynamically decided based on n and BLOCK_SIZE
// Moments shared memory cache size (= (BLOCK_SIZE+4) ^ 2)
#define CACHE_LINE (BLOCK_SIZE + 4)
#define CACHE_SIZE (CACHE_LINE * CACHE_LINE)

// Threads per block (1024 resident blocks / 16 resident threads = 64 threads per block)
#define BLOCK_THREADS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

/* Version 1: Spawn 11 threads per block (not 121 as before), each one taking on 11 moments ie
    thread k takes on moments k, k+11, k+22, ... (11 if we are refering within the block, 
    517 = 47 * 11 = BLOCK_SIZE * GRID_SIZE for matrix G)
    For V2 cache the whole block before starting ops
*/

/* Version 2:
    Note: Currently implemented with just 47 threads
    11x11 grid with 47x47 moments and 47*2=94 threads per block. 10 such resident blocks 
    can fit on a single SM (6 if we used shared memory caching due to memory limitations).
*/

// Cuda kernel function used to calculate one moment per thread
__global__ void cudaKernel(int n, int grid_size, double* gpu_w, int* gpu_G, int* gpu_gTemp){

    // Shared block memory for caching moments (size = (47+4) ^ 2)
    __shared__ int gpu_G_sh[CACHE_SIZE];

	// Moment's coordinates (j->Y, p->X axis) - perform once and save since they are costly
	int x, y;

    // Sum variable to decide what value a moment will take
    double weightSum;

    // Calculate thread_id based on the coordinates of the block
    int blockX = blockIdx.x % grid_size;
    int blockY = blockIdx.x / grid_size;
    int base_id = blockX * BLOCK_SIZE + blockY * n * BLOCK_SIZE;
    //FIX: If grid is not precise, this can get out of bounds
    int thread_id = block_base + threadIdx.x % BLOCK_SIZE + n * (threadIdx.x / BLOCK_SIZE);

    // Indexing variables for caching
    int g_id, sh_x, sh_y, g_x, g_y;

    // Caching G to shared memory
    for(int sh_index = threadIdx.x; sh_index < CACHE_SIZE; sh_index += BLOCK_SIZE){

        // X and Y coordinates on the shared memory
        sh_x = sh_index % CACHE_LINE;
        sh_y = sh_index / CACHE_LINE;

        // Constant starting point (block0 -2, block0 -2) and variable offsets according to thread's current sh_index
        // Calculate coordinates on G based on the shared memory coordinates
        g_x = (blockX * BLOCK_SIZE - 2 + sh_x + n) % n;
        g_y = (blockY * BLOCK_SIZE - 2 + sh_y + n) % n;
        g_id = g_y * n + g_x;

        // Cache G moment to shared memory
        gpu_G_sh[sh_index] = gpu_G[g_id];

    }

    // Sync threads before continuing
    __syncthreads();

	// Check if thread id is within bounds and execute
	if(thread_id < n*n){

        // Iterate through the moments assigned for each thread
        for (int i = thread_id; (i < block_base + n * (BLOCK_SIZE - 1) + BLOCK_SIZE) && (i < n*n); ){
            
            // Calculate moment's coordinates on G (i = y*n + x)
	        x = i % n;
	        y = i / n;

            // Convert these coordinates to relative within the cache block
            x = (x % BLOCK_SIZE) + 2;
            y = (y % BLOCK_SIZE) + 2;

            // Reset weightSum for new moment
            weightSum = 0;

            // Unrolled weights calculations for this moment
            weightSum += gpu_w[0] * gpu_G_sh[(-2 + y ) * CACHE_LINE + (-2 + x)];
            weightSum += gpu_w[1] * gpu_G_sh[(-2 + y ) * CACHE_LINE + (-1 + x)];
            weightSum += gpu_w[2] * gpu_G_sh[(-2 + y ) * CACHE_LINE + x];
            weightSum += gpu_w[3] * gpu_G_sh[(-2 + y ) * CACHE_LINE + (1 + x)];
            weightSum += gpu_w[4] * gpu_G_sh[(-2 + y ) * CACHE_LINE + (2 + x)];
            weightSum += gpu_w[5] * gpu_G_sh[(-1 + y ) * CACHE_LINE + (-2 + x)];
            weightSum += gpu_w[6] * gpu_G_sh[(-1 + y ) * CACHE_LINE + (-1 + x)];
            weightSum += gpu_w[7] * gpu_G_sh[(-1 + y ) * CACHE_LINE + x];
            weightSum += gpu_w[8] * gpu_G_sh[(-1 + y ) * CACHE_LINE + (1 + x)];
            weightSum += gpu_w[9] * gpu_G_sh[(-1 + y ) * CACHE_LINE + (2 + x)];
            weightSum += gpu_w[10] * gpu_G_sh[y * CACHE_LINE + (-2 + x)];
            weightSum += gpu_w[11] * gpu_G_sh[y * CACHE_LINE + (-1 + x)];
            weightSum += gpu_w[13] * gpu_G_sh[y * CACHE_LINE + (1 + x)];
            weightSum += gpu_w[14] * gpu_G_sh[y * CACHE_LINE + (2 + x)];
            weightSum += gpu_w[15] * gpu_G_sh[(1 + y) * CACHE_LINE + (-2 + x)];
            weightSum += gpu_w[16] * gpu_G_sh[(1 + y) * CACHE_LINE + (-1 + x)];
            weightSum += gpu_w[17] * gpu_G_sh[(1 + y) * CACHE_LINE + x];
            weightSum += gpu_w[18] * gpu_G_sh[(1 + y) * CACHE_LINE + (1 + x)];
            weightSum += gpu_w[19] * gpu_G_sh[(1 + y) * CACHE_LINE + (2 + x)];
            weightSum += gpu_w[20] * gpu_G_sh[(2 + y) * CACHE_LINE + (-2 + x)];
            weightSum += gpu_w[21] * gpu_G_sh[(2 + y) * CACHE_LINE + (-1 + x)];
            weightSum += gpu_w[22] * gpu_G_sh[(2 + y) * CACHE_LINE + x];
            weightSum += gpu_w[23] * gpu_G_sh[(2 + y) * CACHE_LINE + (1 + x)];
            weightSum += gpu_w[24] * gpu_G_sh[(2 + y) * CACHE_LINE + (2 + x)];

            // Decide on what future moment should be based on temp:
            // If positive, set to 1. If negative, to -1. If 0, leave untouched
            if(weightSum > 0.0001)
                gpu_gTemp[i] = 1;
            else if(weightSum < -0.0001)
                gpu_gTemp[i] = -1;
            else
                gpu_gTemp[i] = gpu_G[i];

            // Calculate next i
            // Calculate local i and increment by threads number, then calculate new global i
            i = (y % BLOCK_SIZE) * BLOCK_SIZE + (x % BLOCK_SIZE) + BLOCK_THREADS;
            i = block_base + i % BLOCK_SIZE + n * (i / BLOCK_SIZE);
        }
	}
}

void printResult(int *G, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", G[i*n+j]);
        }
        printf("\n");
    }

}

void ising( int *G, double *w, int k, int n){

	// Calculate number of blocks
    int grid_size = ((n % BLOCK_SIZE) == 0) ? (n/BLOCK_SIZE) : (n/BLOCK_SIZE + 1);

	// Use cuda memcpy to copy weights array w to gpu memory gpu_w
	double *gpu_w; 
	cudaMalloc(&gpu_w, 25*sizeof(double));
	cudaMemcpy(gpu_w, w, 25*sizeof(double), cudaMemcpyHostToDevice);
	
	// Array to hold the data of G in GPU memory
	int *gpu_G;
	cudaMalloc(&gpu_G, n*n*sizeof(int));
	cudaMemcpy(gpu_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);

	// Temporary GPU memory array used as the modified one in tandem with gpu_G
	int *gpu_gTemp;
	cudaMalloc(&gpu_gTemp, n*n*sizeof(int));

	// Temporary pointer used for swapping gpu_G and gpu_gTemp
	int *gpu_swapPtr;

	// Define grid and block dimensions - avoid using dim objects for now
	//dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Call cudaKernel for each iteration using pointers to cuda memory
		cudaKernel<<<grid_size*grid_size, BLOCK_SIZE>>>(n, gpu_w, gpu_G, gpu_gTemp);

		// Synchronize threads before swapping pointers
		cudaDeviceSynchronize();

		// Swap gpu_G and gpu_gTemp pointers for next iteration to avoid copying data on every iteration
		gpu_swapPtr = gpu_G;
		gpu_G = gpu_gTemp;
		gpu_gTemp = gpu_swapPtr;
	}

	// Copy final data to CPU memory
    cudaMemcpy(G, gpu_G, n*n*sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(gpu_G);
	cudaFree(gpu_gTemp);

}

int main(){

	// Set dimensions and number of iterations
	int n = 517;	int k = 1;

	// Define weights array
    double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
    		0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};

	// Open binary file and write contents to an array
    FILE *fptr = fopen("conf-init.bin","rb");
    int *G = (int*)scalloc(n*n, sizeof(int));
    if (fptr == NULL){
        printf("Error! opening file");
        exit(1);
    }
    fread(G, sizeof(int), n*n, fptr);
	fclose(fptr);

    // Call ising
    ising(G, weights, k, n);

	// Open results binary file and write contents to an array
    FILE *fptrR = fopen("conf-1.bin","rb");
    int *R = (int*)scalloc(n*n, sizeof(int));
    if (fptrR == NULL){
        printf("Error! opening file");
        exit(1);
    }
    fread(R, sizeof(int), n*n, fptr);
	fclose(fptrR);

	// Check results
	int errNum = 0;
    for (int i=0; i < n*n; i++)
		if(G[i] != R[i])
			errNum++;
	printf("Done testing, found %d errors", errNum);

    return 0;
}