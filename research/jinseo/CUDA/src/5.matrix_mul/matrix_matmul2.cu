/*****************************************************************************
 * File:        matrixMul.cu
 * Description: Matrix multiplication, C = AB
 *              A has m x k dimensions, B has k x n dimensions, and C has
 *              m x n dimensions.
 *              It is not for the most performance.
 *              
 * Compile:     nvcc -o matrixMul matrixMul.cu -I.. -lcuda
 * Run:         ./matrixMul <m> <k> <n>
 *                  <m> : the number of rows in Matrix A
 *                  <k> : the number of columns in Matrix A, it is also
 *                        the number of rows in Matrix B.
 *                  <n> : the number of columns in Matrix B.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

const int RMAX = 1000000;

void Usage(char prog_name[]);
__global__ void matrixMul(const float *A, const float *B, float *C, const int M, const int K, const int N, const int
    threadnum);

int main(int argc, char* argv[])
{
    if (argc != 5) {
        Usage(argv[0]);
    }

    int m = strtol(argv[1], NULL, 10);
    int k = strtol(argv[2], NULL, 10);
    int n = strtol(argv[3], NULL, 10);
    long threadnum = strtol(argv[4], NULL, 10);
    printf("[Matrix multiplication, C = AB]\n");
    printf("\tA is (%d x %d) matrix, B is (%d x %d) matrix, and \n", m, k, k, n);
    printf("\tC is (%d x %d) matrix.\n", m, n);

    // Allocate the host matrix A, B, C
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_SUCCESS);
    }
    
    // Initialize that host matrix
    common_random_init_matrix<float>(h_A, m, k);
    common_random_init_matrix<float>(h_B, k, n);

    // Allocate the device matrix A, B, C
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy the host input matrix A and B in host memory 
    // to the device input matrix in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate CUDA events for estimating
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch the Matrix Multiplication CUDA Kernel
    const int block_size = 1;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    matrixMul<<<1, threadnum>>>(d_A, d_B, d_C, m, k, n, threadnum);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
   // Copy the device result matrix in device memory
    // to the host result matrix in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify that the result matrix is correct (L2-norm error)
// common_verify_matMul_l2ne(h_A, h_B, h_C, m, k, n);
    
    // Compute and Print the performance
    float msecTotal = 0.0f; 
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop)); 
    printf("Execution time is .... %.3f msec\n", msecTotal);
    
    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <m> <k> <n>\n", prog_name);
    fprintf(stderr, "\t<m> : the number of rows in matrix A.\n");
    fprintf(stderr, "\t<k> : the number of columns in Matrix A, it is also\n");
    fprintf(stderr, "\t      the number of rows in Matrix B.\n");
    fprintf(stderr, "\t<n> : the number of columns in matrix B.\n");
    exit(EXIT_FAILURE);
}

void Generate_matrix(float mat[], int m, int n){
  for(int i =0; i < m; i++){
    for(int j = 0; j < n; j++)
      mat[i*n + j] = (rand() % RMAX) / (RMAX /10.0);
  }
}
__global__
void matrixMul(const float *A, const float *B, float *C, const int M, const int K, const int N, const int threadnum)
{
    int my_rank = threadIdx.x;
    int local_m = M/threadnum;
    int first_row = my_rank * local_m;
    int last_row = first_row + local_m;

    float temp;
    for(int i = first_row; i < last_row; i++){
      for(int j = 0; j < K; j++){
        temp = 0.0;
        for(int l = 0; l < N; l++){
          temp += A[i*N + l] * B[l*K + j];
        }
        C[i*K + j] = temp;
      }
    }
  
}
