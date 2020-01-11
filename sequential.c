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

// Ising model evolution function
void ising( int *G, double *w, int k, int n){
	
	// Array to hold local copy of G to freely modify the original
	int *gPrev = calloc(n*n, sizeof(int));

	// Temporary variable to decide what value a moment will take
	double temp = 0;

	// Variables to hold final indices when examining each moment
	int indX, indY;

	// Iterate the number of desired times
	for(int i = 0; i < k; i++){

		// Make a copy of G as it is when beginning the evolution process
		memcpy(gPrev, G, n*n*sizeof(int));

		// Iterate for every moment of G (y->Y, x->X axis)
		for(int y = 0; y < n; y++){
			for(int x = 0; x < n; x++){

				// Reset temporary variable
				temp = 0;

				// Iterate through the moment's 5x5 neighborhood (l->Y, m->X axis)
				for(int l = 0; l < 5; l++){
					for(int m = 0; m < 5; m++){

						// Skip examining the point itself
						if((l == 2) && (m == 2))
							continue;

						// Decide wrap-around neighbor indexes - 2 is subtracted to center the neighbors grid on the moment
						// Check for negatives (underflow) and positives over n (overflow)
						indY = ((l-2) + y + n) % n;
						indX = ((m-2) + x + n) % n;
						
						// Add to temp the weight*value of the original neighbor
						temp += w[l * 5 + m] * gPrev[indY * n + indX];

					}
				}

				// Decide on what future moment should be based on temp:
				// If positive, set to 1. If negative, to -1. If 0, leave untouched
				if(temp > 0.0001)
					G[y * n + x] = 1;
				else if(temp < -0.0001)
					G[y * n + x] = -1;
				else
					G[y * n + x] = G[y * n + x];

			}
		}

	}
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

    /* Enable to use with input binaries
	// Open binary file and write contents to an array
    FILE *fptr = fopen("conf-init.bin","rb");
    int *G = (int*)calloc(n*n, sizeof(int));
    if (fptr == NULL){
        printf("Error! opening file");
        exit(1);
    }
    fread(G, sizeof(int), n*n, fptr);
    fclose(fptr);
    */

    // Generate random input data
    srand(time(NULL));
    int *G = calloc(n*n, sizeof(int));
    for(int i = 0; i < n*n; i++)
        G[i] = rand() % 2;

    // Call ising model evolution function
    ising(G, weights, k, n);

    /* Enable to use with input binaries
	// Open results binary file and write contents to an array
    FILE *fptrR = fopen("conf-1.bin","rb");
    int *R = (int*)calloc(n*n, sizeof(int));
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
    
    */

    return 0;
}