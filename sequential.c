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

		// Iterate for every moment of G (j->Y, p->X axis)
		for(int j = 0; j < n; j++){
			for(int p = 0; p < n; p++){

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
						indY = ((l-2) + j + n) % n;
						indX = ((m-2) + p + n) % n;
						
						// Add to temp the weight*value of the original neighbor
						temp += w[l * 5 + m] * gPrev[indY * n + indX];

					}
				}

				// Decide on what future moment should be based on temp:
				// If positive, set to 1. If negative, to -1. If 0, leave untouched
				if(temp > 0.0001)
					G[j * n + p] = 1;
				else if(temp < -0.0001)
					G[j * n + p] = -1;
                		else
                    			G[j * n + p] = G[j * n + p];

			}
		}

	}
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