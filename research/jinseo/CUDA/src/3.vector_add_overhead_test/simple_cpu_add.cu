#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 8192


void add(float *a, float *b, float *c, float size){
  for(int i = 0; i < size; i++){
    c[i] = a[i] +b[i];
  }
 }



void random_floats(float* x, int size)
{
	int i;
  float max = 10000.0;
	for (i=0;i<size;i++) {
		x[i]=((float)rand()/(float)(RAND_MAX))*max;
	}
}

int main(void) 
{
	float *a, *b, *c; // host copies of a, b, c
	int size = N * sizeof(float);
	
	
	// Alloc space for host copies of a, b, c and setup input values
	a = (float *)malloc(size); random_floats(a, N);
	b = (float *)malloc(size); random_floats(b, N);
	c = (float *)malloc(size);

	// Copy inputs to device
	
	// Launch add() kernel on GPU with N blocks
  add(a, b, c, N);
	for (int i=0;i<N;i++) {
		printf("a[%d]=%f , b[%d]=%f, c[%d]=%f\n",i,a[i],i,b[i],i,c[i]);
	}

	free(a); free(b); free(c);
	return 0;
}
