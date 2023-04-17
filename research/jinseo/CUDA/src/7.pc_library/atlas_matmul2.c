#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

#define GET_TIME(now){ \
  struct timeval t; \
  gettimeofday(&t, NULL); \
  now = t.tv_sec + t.tv_usec/10000000.0; \
}

const int RMAX = 1000000;
int m, n, k, sol;

void Generate_matrix(double *mat, int m, int n);
void Get_args(int argc, char* argv[]);
void Usage(char* prog_name);

int main (int argc, char* argv[])
{
  double *A, *B, *C;
  printf("starttttt");
  Get_args(argc, argv);
  printf("log\n");

  A = (double*)malloc(sizeof(double)*n*m);
  B = (double*)malloc(sizeof(double)*n*k);
  C = (double*)malloc(sizeof(double)*m*k);
  printf("log\n");
  Generate_matrix(A, m, n);
  printf("log\n");

  Generate_matrix(B, n, k);
  printf("log\n");

  double start, finish;
  GET_TIME(start);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A, n, B, k, 0, C, k);
  GET_TIME(finish);
  printf("Elapsed time = %.6f seconds\n", finish-start);

 
  free(A);
  free(B);
  free(C);

  return 0;
}

void Get_args(int argc, char* argv[]){
  if(argc != 4)
    Usage(argv[0]);
  m = strtol(argv[1], NULL, 10);
  n = strtol(argv[2], NULL, 10);
  k = strtol(argv[3], NULL, 10);
  if ( m <= 0 || n <= 0 || k <= 0)
    Usage(argv[0]);
}
void Usage(char* prog_name){
    fprintf(stderr, "Usage: %s <m> <n> <k> <sol>\n", prog_name);
    exit(0);
}



void Generate_matrix(double *mat, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          mat[i*n+j] = (rand() % RMAX) / (RMAX / 10.0);
}


