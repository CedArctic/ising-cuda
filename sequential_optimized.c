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

void printResult(int *G, int n){
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

	// Variables to hold final indices when examining each moment
	int indX, indY;

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Iterate for every moment of G (j->Y, p->X axis)
		for(int j = 0; j < n; j++){
			for(int p = 0; p < n; p++){

				// Reset temporary variable
				weightSum = 0;

				// Iterate through the moment's 5x5 neighborhood (l->Y, m->X axis)
				for(int l = 0; l < 5; l++){
					for(int m = 0; m < 5; m++){

						// Skip examining the point itself
						if((l == 2) && (m == 2))
							continue;

						// Decide wrap-around neighbor indexes - 2 is subtracted to center the neighbors grid on the moment
						// Check for negatives (underflow) and positives over n (overflow)
						indY = ((l-2) + j + n) % n;
						indX = ((m-2) + p + n) % n;
						
						// Add to temp the weight*value of the original neighbor
						weightSum += w[l * 5 + m] * G[indY * n + indX];

					}
				}

				// Decide on what future moment should be based on temp:
				// If positive, set to 1. If negative, to -1. If 0, leave untouched
				if(weightSum > 0.0001)
					gTemp[j * n + p] = 1;
				else if(weightSum < -0.0001)
					gTemp[j * n + p] = -1;
				else
					gTemp[j * n + p] = G[j * n + p];

			}
		}

		// Swap G and gTemp pointers for next iteration to avoid copying data on every iteration
		swapPtr = G;
		G = gTemp;
		gTemp = swapPtr;
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

	// Open binary file and write contents to an array
    FILE *fptr = fopen("conf-init.bin","rb");
    printf("Pointer created\n");
    int *G = calloc(n*n, sizeof(int));
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

