#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000

__global__ void add(float *a, float *b, float *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
  clock_t start1 = clock();
	float *a, *b, *c;
	float *d_a, *d_b, *d_c; 
  int flt_num = 10000;
	int size =  flt_num * sizeof(float);

	clock_t start2 = clock();
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	clock_t end2 = clock();

  clock_t start3 = clock();
	a = (float *)malloc(size); 
  random_floats(a, flt_num);
	b = (float *)malloc(size); 
  random_floats(b, flt_num);
	c = (float *)malloc(size);
  clock_t end3 = clock();

  clock_t start4 = clock();
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	clock_t end4 = clock();

  clock_t start5 = clock();
	add<<<N,4>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  clock_t end5 = clock();

	for (int i=0;i<flt_num;i++) {
		printf("a[%d]=%f , b[%d]=%f, c[%d]=%f\n",i,a[i],i,b[i],i,c[i]);
	}

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  clock_t end = clock();
  printf("Device malloc time: %.3f msec\n", (double)(end2 - start2) / CLOCKS_PER_SEC*1000);
 // printf("Host malloc & float generation time: %.3f msec\n", (double)(end3 - start3) / CLOCKS_PER_SEC*1000);
  printf("Device memcpy  time: %lf msec\n", (double)(end4 - start4) / CLOCKS_PER_SEC*1000);
  printf("Device execution & return time: %.3f msec\n", (double)(end5 - start5) / CLOCKS_PER_SEC*1000);
  printf("Execution time: %.3f msec\n", (double)(end - start1) / CLOCKS_PER_SEC*1000);

      

	return 0;
}
