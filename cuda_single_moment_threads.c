// Block size axis (BLOCK_SIZE^2 = number of threads per block)
#define BLOCK_SIZE 11
// Number of blocks on axis (GRID_SIZE^2 = number of blocks in grid)
//#define GRID_SIZE 47 // Number is now dynamically decided based on n and BLOCK_SIZE

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

// Cuda kernel function used to calculate one moment per thread
__global__ void cudaKernel(int n, double* gpu_w, int* gpu_G, int* gpu_gTemp){

	// Calculate thread id
	int thread_id = blockIdx.x * BLOCK_SIZE * BLOCK_SIZE + threadIdx.x;

	// Moment's coordinates (i = y*n + x) - perform once and save since they are costly
	int x = thread_id % n;
	int y = thread_id / n;
	
	// Sum variable to decide what value a moment will take
	double weightSum = 0;

	// Check if thread id is within bounds and execute
	if(thread_id < n*n){

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
			gpu_gTemp[thread_id] = 1;
		else if(weightSum < -0.0001)
			gpu_gTemp[thread_id] = -1;
		else
			gpu_gTemp[thread_id] = gpu_G[thread_id];
	}
}

// Cuda kernel function used to check for early exit if G == gTemp
__global__ void exitKernel(int n, int* gpu_G, int* gpu_gTemp, int* gpu_exitFlag){
	
	// Shared block exit flag
    __shared__ int blockFlag = 0;
	
	// Calculate thread id
	int thread_id = blockIdx.x * BLOCK_SIZE * BLOCK_SIZE + threadIdx.x;
	
	// If two values are not the same, increment the flag
	// This is not race-condition safe but we don't care since one write is guaranteed to finish
	if(gpu_gTemp[thread_id] == gpu_G[thread_id])
		atomicAdd(&blockFlag, 1);
	
	// Sync threads before writing to global
	__syncthreads();
	
	// First thread of the block writes flag back to the global memory
	if((thread_id == blockIdx.x * BLOCK_SIZE * BLOCK_SIZE) && (blockFlag > 0))
		atomicAdd(gpu_exitFlag, blockFlag);
	
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

	// Calculate number of blocks in a 2D grid with a predefined block size
	//int blocks = ((n*n) % (BLOCK_SIZE * BLOCK_SIZE) == 0) ? ((n*n) / (BLOCK_SIZE * BLOCK_SIZE)) : ((n*n) / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
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
	
	// GPU early exit flag
	int *gpu_exitFlag;
	int exitFlag = 0;
	cudaMalloc(&gpu_exitFlag, sizeof(int));
	cudaMemcpy(gpu_exitFlag, &exitFlag, sizeof(int), cudaMemcpyHostToDevice);

	// Define grid and block dimensions - disabled for now
	//dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Call cudaKernel for each iteration using pointers to cuda memory
		cudaKernel<<<grid_size * grid_size, BLOCK_SIZE*BLOCK_SIZE>>>(n, gpu_w, gpu_G, gpu_gTemp);

		// Synchronize threads before swapping pointers
		cudaDeviceSynchronize();

		// Swap gpu_G and gpu_gTemp pointers for next iteration to avoid copying data on every iteration
		gpu_swapPtr = gpu_G;
		gpu_G = gpu_gTemp;
		gpu_gTemp = gpu_swapPtr;
		
		// Check for early exit
		exitKernel<<<grid_size * grid_size, BLOCK_SIZE*BLOCK_SIZE>>>(n, gpu_G, gpu_gTemp, gpu_exitFlag);
		cudaDeviceSynchronize();
		cudaMemcpy(&exitFlag, gpu_exitFlag, sizeof(int), cudaMemcpyDeviceToHost);
		if(exitFlag > 0)
			break;
		
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

