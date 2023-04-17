#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define N 32

__global__ void add(float *a, float *b, float *c, int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements){
    c[i] = a[i] + b[i];
  }
 // printf("in GPU Calculation Result is... %.3f\n",c[blockIdx.x]);
 // printf("a=%f , b=%f, c=%f\n",a[blockIdx.x],b[blockIdx.x],c[blockIdx.x]);

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
{ clock_t start, end;
  start = clock();
	float *a, *b, *c;
	float *d_a, *d_b, *d_c; 
  int flt_num = 16384;
	int size =  flt_num * sizeof(float);

  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

  cudaEventRecord(start1);
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
	a = (float *)malloc(size); 
  random_floats(a, flt_num);
	b = (float *)malloc(size); 
  random_floats(b, flt_num);
	c = (float *)malloc(size);

  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);

  cudaEventRecord(start2);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop2);
  cudaEventSynchronize(stop2);

  cudaEvent_t start3, stop3;
  cudaEventCreate(&start3);
  cudaEventCreate(&stop3);

  cudaEventRecord(start3);
	add<<<1,16384>>>(d_a, d_b, d_c, 16384);

  cudaDeviceSynchronize();
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop3);
  cudaEventSynchronize(stop3);

	for (int i=0;i<flt_num;i++) {
		printf("a[%d]=%f , b[%d]=%f, c[%d]=%f\n",i,a[i],i,b[i],i,c[i]);
	}

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  float msec_malloc = 0.0f;
  float msec_memcpy = 0.0f;
  float msec_execution = 0.0f;
  cudaEventElapsedTime(&msec_malloc, start1, stop1);
  cudaEventElapsedTime(&msec_memcpy, start2, stop2);
  cudaEventElapsedTime(&msec_execution, start3, stop3);
  printf("Device malloc time : %.3f msec\n" ,msec_malloc);
  printf("Device memcpy time : %.3f msec\n", msec_memcpy);
  printf("Device execution time : %.3f msec\n", msec_execution);
  end = clock();
  printf("Execution time: %.3f msec\n", (double)(end - start) / CLOCKS_PER_SEC*1000);
      

	return 0;
}
