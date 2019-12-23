#define DIMENSIONS 517
#define WEIGHTS_DIM 5

#include <stdio.h>
#include <string.h>

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
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

		// Iterate for every moment of G (j->X, k->Y axis)
		for(int j = 0; j < n; j++){
			for(int k = 0; k < n; k++){

				// Reset temporary variable
				temp = 0;

				// Iterate through the moment's 5x5 neighborhood (l->X, m->Y axis)
				for(int l = 0; l < 5; l++){
					for(int m = 0; m < 5; m++){

						// Skip examining the point itself
						if((l == 3) && (m==3))
							continue;

						// Decide wrap-around neighbor indexes - 2 is subtracted to center the neighbors grid on the moment

						// Check for negatives
						indX = ((l-2) + j > 0)?(l-2) + j:n-((l-2) + j);
						indY = ((m-2) + k > 0)?(m-2) + k:n-((m-2) + k);

						// Check for over n
						indX = indX % n;
						indY = indY % n;

						// Add to temp the weight*value of the original neighbor
						temp += w[l * n + m] * gPrev[indX * n + indY];

					}
				}

				// Decide on what future moment should be based on temp:
				// If positive, set to 1. If negative, to -1. If 0, leave untouched
				if(temp > 0)
					G[i * n + k] = 1;
				if(temp < 0)
					G[i * n + k] = 1;

			}
		}

	}

}

int main(){

	// Set dimensions and number of iterations
	int n = DIMENSIONS;	int k = 1;

	// Open binary file and write contents to an array
    FILE *fptr = fopen("conf-init.bin","rb");
    int G[DIMENSIONS][DIMENSIONS];
    if (fptr == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    fread(&G, sizeof(int), DIMENSIONS*DIMENSIONS, fptr);

    // Define weights array
    double weights = {
    		{0.004, 0.016, 0.026, 0.016, 0.004},
    		{0.016, 0.071, 0.117, 0.071, 0.016},
    		{0.026, 0.117, 0, 0.117, 0.026},
    		{0.016, 0.071, 0.117, 0.071, 0.016},
    		{0.004, 0.016, 0.026, 0.016, 0.004}
    };

    // Call

    // Close binary file
    fclose(fptr);
    printf("Done");

    return 0;
}
