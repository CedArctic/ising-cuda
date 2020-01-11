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

// Function to print out the moments matrix
void printMoments(int *G, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", G[i*n+j]);
        }
        printf("\n");
    }

}

void ising( int *G, double *w, int k, int n){
	
	// Temporary array used as the modified one in tandem with G
	int *gTemp = calloc(n*n, sizeof(int));

	// Temporary pointer used for swapping G and gTemp
	int *swapPtr;

	// Sum variable to decide what value a moment will take
	double weightSum = 0;

	int j,p;

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Iterate for every moment of G (j->Y, p->X axis)
		for(int f = 0; f < n*n; f++){

			p = f % n;
			j = f / n;

			// Reset temporary variable
			weightSum = 0;

			// Unrolled weights calculations for this moment
			weightSum += w[0] * G[((-2 + j + n) % n) * n + (-2 + p + n) % n];
			weightSum += w[1] * G[((-2 + j + n) % n) * n + (-1 + p + n) % n];
			weightSum += w[2] * G[((-2 + j + n) % n) * n + (p + n) % n];
			weightSum += w[3] * G[((-2 + j + n) % n) * n + (1 + p + n) % n];
			weightSum += w[4] * G[((-2 + j + n) % n) * n + (2 + p + n) % n];
			weightSum += w[5] * G[((-1 + j + n) % n) * n + (-2 + p + n) % n];
			weightSum += w[6] * G[((-1 + j + n) % n) * n + (-1 + p + n) % n];
			weightSum += w[7] * G[((-1 + j + n) % n) * n + (p + n) % n];
			weightSum += w[8] * G[((-1 + j + n) % n) * n + (1 + p + n) % n];
			weightSum += w[9] * G[((-1 + j + n) % n) * n + (2 + p + n) % n];
			weightSum += w[10] * G[((j + n) % n) * n + (-2 + p + n) % n];
			weightSum += w[11] * G[((j + n) % n) * n + (-1 + p + n) % n];
			weightSum += w[13] * G[((j + n) % n) * n + (1 + p + n) % n];
			weightSum += w[14] * G[((j + n) % n) * n + (2 + p + n) % n];
			weightSum += w[15] * G[((1 + j + n) % n) * n + (-2 + p + n) % n];
			weightSum += w[16] * G[((1 + j + n) % n) * n + (-1 + p + n) % n];
			weightSum += w[17] * G[((1 + j + n) % n) * n + (p + n) % n];
			weightSum += w[18] * G[((1 + j + n) % n) * n + (1 + p + n) % n];
			weightSum += w[19] * G[((1 + j + n) % n) * n + (2 + p + n) % n];
			weightSum += w[20] * G[((2 + j + n) % n) * n + (-2 + p + n) % n];
			weightSum += w[21] * G[((2 + j + n) % n) * n + (-1 + p + n) % n];
			weightSum += w[22] * G[((2 + j + n) % n) * n + (p + n) % n];
			weightSum += w[23] * G[((2 + j + n) % n) * n + (1 + p + n) % n];
			weightSum += w[24] * G[((2 + j + n) % n) * n + (2 + p + n) % n];

			// Decide on what future moment should be based on temp:
			// If positive, set to 1. If negative, to -1. If 0, leave untouched
			if(weightSum > 0.0001)
				gTemp[j * n + p] = 1;
			else if(weightSum < -0.0001)
				gTemp[j * n + p] = -1;
			else
				gTemp[j * n + p] = G[j * n + p];
		}

		// Swap G and gTemp pointers for next iteration to avoid copying data on every iteration
		swapPtr = G;
		G = gTemp;
		gTemp = swapPtr;
		
		// Check if gTemp == G - if so, no point in performing more iterations
		// Use weightSum as a boolean variable to avoid declaring another variable
		weightSum = 0;
		for(int k = 0; k < n*n; k++){
			if(gTemp[k] != G[k]){
				weightSum = 1;
				break;
			}
		}
		if(weightSum != 1)
			break;
		
	}

	/* If final k is odd, the current data is in the original gTemp's memory space where G (pointer) 
	is pointing now due to the swap performed above. So we need to copy the data from gTemp's space 
	(current G pointer) to G's space (current gTemp pointer)
	*/
	if(k % 2 == 1)
		memcpy(gTemp, G, n*n*sizeof(int));

}

int main(){

	// Set dimensions and number of iterations
	int n = 517;	int k = 1;

	// Open binary file, write contents to an array and close it
    FILE *fptr = fopen("conf-init.bin","rb");
    if (fptr == NULL){
        printf("Error opening file");
        // Program exits if the file pointer returnscalloc(n*n, sizeof(int)); NULL.
        exit(1);
    }
	int *G = calloc(n*n, sizeof(int));
    fread(G, sizeof(int), n*n, fptr);
    fclose(fptr);

    // Define weights array
    double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
    		0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};

    // Call ising
    ising(G, weights, k, n);

	// Check results by comparing with ready data

	// Check for k = 1
	int *expected = calloc(n*n, sizeof(int));
	fptr = fopen("conf-1.bin","rb");
	fread(expected, sizeof(int), n*n, fptr);
	fclose(fptr);
	for(int v = 0; v < n*n; v++)
		if(expected[v] != G[v])
			printf("Error on test k=1\n");
		
	// Check for k = 4
	fptr = fopen("conf-4.bin","rb");
	fread(expected, sizeof(int), n*n, fptr);
	fclose(fptr);
	for(int v = 0; v < n*n; v++)
		if(expected[v] != G[v])
			printf("Error on test k=4\n");
	
	// Check for k = 11
	fptr = fopen("conf-11.bin","rb");
	fread(expected, sizeof(int), n*n, fptr);
	fclose(fptr);
	for(int v = 0; v < n*n; v++)
		if(expected[v] != G[v])
			printf("Error on test k=11\n");

    return 0;
}

