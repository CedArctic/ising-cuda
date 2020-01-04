// Block size axis (BLOCK_SIZE^2 = number of threads per block)
#define BLOCK_SIZE 11
// Number of blocks on axis (GRID_SIZE^2 = number of blocks in grid)
#define GRID_SIZE 47

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
		weightSum += gxu_w[0] * gxu_G[((-2 + y + n) % n) * n + (-2 + x + n) % n];
		weightSum += gxu_w[1] * gxu_G[((-2 + y + n) % n) * n + (-1 + x + n) % n];
		weightSum += gxu_w[2] * gxu_G[((-2 + y + n) % n) * n + (x + n) % n];
		weightSum += gxu_w[3] * gxu_G[((-2 + y + n) % n) * n + (1 + x + n) % n];
		weightSum += gxu_w[4] * gxu_G[((-2 + y + n) % n) * n + (2 + x + n) % n];
		weightSum += gxu_w[5] * gxu_G[((-1 + y + n) % n) * n + (-2 + x + n) % n];
		weightSum += gxu_w[6] * gxu_G[((-1 + y + n) % n) * n + (-1 + x + n) % n];
		weightSum += gxu_w[7] * gxu_G[((-1 + y + n) % n) * n + (x + n) % n];
		weightSum += gxu_w[8] * gxu_G[((-1 + y + n) % n) * n + (1 + x + n) % n];
		weightSum += gxu_w[9] * gxu_G[((-1 + y + n) % n) * n + (2 + x + n) % n];
		weightSum += gxu_w[10] * gxu_G[((y + n) % n) * n + (-2 + x + n) % n];
		weightSum += gxu_w[11] * gxu_G[((y + n) % n) * n + (-1 + x + n) % n];
		weightSum += gxu_w[13] * gxu_G[((y + n) % n) * n + (1 + x + n) % n];
		weightSum += gxu_w[14] * gxu_G[((y + n) % n) * n + (2 + x + n) % n];
		weightSum += gxu_w[15] * gxu_G[((1 + y + n) % n) * n + (-2 + x + n) % n];
		weightSum += gxu_w[16] * gxu_G[((1 + y + n) % n) * n + (-1 + x + n) % n];
		weightSum += gxu_w[17] * gxu_G[((1 + y + n) % n) * n + (x + n) % n];
		weightSum += gxu_w[18] * gxu_G[((1 + y + n) % n) * n + (1 + x + n) % n];
		weightSum += gxu_w[19] * gxu_G[((1 + y + n) % n) * n + (2 + x + n) % n];
		weightSum += gxu_w[20] * gxu_G[((2 + y + n) % n) * n + (-2 + x + n) % n];
		weightSum += gxu_w[21] * gxu_G[((2 + y + n) % n) * n + (-1 + x + n) % n];
		weightSum += gxu_w[22] * gxu_G[((2 + y + n) % n) * n + (x + n) % n];
		weightSum += gxu_w[23] * gxu_G[((2 + y + n) % n) * n + (1 + x + n) % n];
		weightSum += gxu_w[24] * gxu_G[((2 + y + n) % n) * n + (2 + x + n) % n];

		// Decide on what future moment should be based on temp:
		// If positive, set to 1. If negative, to -1. If 0, leave untouched
		if(weightSum > 0.0001)
			gpu_gTemp[i] = 1;
		else if(weightSum < -0.0001)
			gpu_gTemp[i] = -1;
		else
			gpu_gTemp[i] = gpu_G[i];
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
	//int blocks = ((n*n) % BLOCK_SIZE == 0) ? ((n*n) / BLOCK_SIZE) : ((n*n) / BLOCK_SIZE) + 1;

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

	// Define grid and block dimensions - disabled for now
	//dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Call cudaKernel for each iteration using pointers to cuda memory
		cudaKernel<<<GRID_SIZE*GRID_SIZE, BLOCK_SIZE*BLOCK_SIZE>>>(n, gpu_w, gpu_G, gpu_gTemp);

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
    int *G = (int*)scalloc(n*n, sizeof(int));
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

