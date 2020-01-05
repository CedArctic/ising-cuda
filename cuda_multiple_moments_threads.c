// Block size axis (BLOCK_SIZE^2 = number of threads per block)
#define BLOCK_SIZE 47
// Number of blocks on axis (GRID_SIZE^2 = number of blocks in grid)
//#define GRID_SIZE 11  // Number is now dynamically decided based on n and BLOCK_SIZE

// Threads per block
#define BLOCK_THREADS 256

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

/* Version 3:
    Blocks per thread are independent of block size
*/

// Cuda kernel function used to calculate one moment per thread
__global__ void cudaKernel(int n, int grid_size, double* gpu_w, int* gpu_G, int* gpu_gTemp){

	// Moment's coordinates
	int x, y;

    // Sum variable to decide what value a moment will take
    double weightSum;

    // Calculate thread_id based on the coordinates of the block
    int blockX = blockIdx.x % grid_size;
    int blockY = blockIdx.x / grid_size;
    int block_base = blockX * BLOCK_SIZE + blockY * n * BLOCK_SIZE;
    int thread_id = block_base + threadIdx.x % BLOCK_SIZE + n * (threadIdx.x / BLOCK_SIZE);

	// Check if thread id is within bounds and execute
	if(thread_id < n*n){

        // Iterate through the moments assigned for each thread
        for (int i = thread_id; (i <= block_base + n * (BLOCK_SIZE - 1) + BLOCK_SIZE) && (i < n*n); ){
            
            // Calculate moment's coordinates (i = y*n + x)
	        x = i % n;
	        y = i / n;

            // Reset weightSum for new moment
            weightSum = 0;

            // Unrolled weights calculations for this moment
            weightSum += gpu_w[0] * gpu_G[((-2 + y + n) % n) * n + (-2 + x + n) % n];
            weightSum += gpu_w[1] * gpu_G[((-2 + y + n) % n) * n + (-1 + x + n) % n];
            weightSum += gpu_w[2] * gpu_G[((-2 + y + n) % n) * n + (x + n) % n];
            weightSum += gpu_w[3] * gpu_G[((-2 + y + n) % n) * n + (1 + x + n) % n];
            weightSum += gpu_w[4] * gpu_G[((-2 + y + n) % n) * n + (2 + x + n) % n];
            weightSum += gpu_w[5] * gpu_G[((-1 + y + n) % n) * n + (-2 + x + n) % n];
            weightSum += gpu_w[6] * gpu_G[((-1 + y + n) % n) * n + (-1 + x + n) % n];
            weightSum += gpu_w[7] * gpu_G[((-1 + y + n) % n) * n + (x + n) % n];
            weightSum += gpu_w[8] * gpu_G[((-1 + y + n) % n) * n + (1 + x + n) % n];
            weightSum += gpu_w[9] * gpu_G[((-1 + y + n) % n) * n + (2 + x + n) % n];
            weightSum += gpu_w[10] * gpu_G[((y + n) % n) * n + (-2 + x + n) % n];
            weightSum += gpu_w[11] * gpu_G[((y + n) % n) * n + (-1 + x + n) % n];
            weightSum += gpu_w[13] * gpu_G[((y + n) % n) * n + (1 + x + n) % n];
            weightSum += gpu_w[14] * gpu_G[((y + n) % n) * n + (2 + x + n) % n];
            weightSum += gpu_w[15] * gpu_G[((1 + y + n) % n) * n + (-2 + x + n) % n];
            weightSum += gpu_w[16] * gpu_G[((1 + y + n) % n) * n + (-1 + x + n) % n];
            weightSum += gpu_w[17] * gpu_G[((1 + y + n) % n) * n + (x + n) % n];
            weightSum += gpu_w[18] * gpu_G[((1 + y + n) % n) * n + (1 + x + n) % n];
            weightSum += gpu_w[19] * gpu_G[((1 + y + n) % n) * n + (2 + x + n) % n];
            weightSum += gpu_w[20] * gpu_G[((2 + y + n) % n) * n + (-2 + x + n) % n];
            weightSum += gpu_w[21] * gpu_G[((2 + y + n) % n) * n + (-1 + x + n) % n];
            weightSum += gpu_w[22] * gpu_G[((2 + y + n) % n) * n + (x + n) % n];
            weightSum += gpu_w[23] * gpu_G[((2 + y + n) % n) * n + (1 + x + n) % n];
            weightSum += gpu_w[24] * gpu_G[((2 + y + n) % n) * n + (2 + x + n) % n];

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
		cudaKernel<<<grid_size*grid_size, BLOCK_SIZE>>>(n, grid_size, gpu_w, gpu_G, gpu_gTemp);

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

	// Open binary file and write contents to an array
    FILE *fptr = fopen("conf-init.bin","rb");
    printf("Pointer created\n");
    int *G = (int*)calloc(n*n, sizeof(int));
    printf("G allocated\n");
    if (fptr == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }

    fread(G, sizeof(int), n*n, fptr);

    // Define weights array
    double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
    		0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};

    // Call ising
    ising(G, weights, k, n);

    // Close binary file
    fclose(fptr);
    printf("Done");

    return 0;
}